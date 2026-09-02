//! The progress bar for `ferx bootstrap`.
//!
//! A 200-sample bootstrap is minutes to hours of fitting behind one line of
//! output, and until it finished there was nothing on the terminal to say
//! whether it was making progress or wedged. This renders
//! [`ferx_tools::bootstrap::BootstrapEvent`]s onto an `indicatif` bar.
//!
//! Everything here is presentation: the events themselves carry no terminal
//! concern, so the R package renders the same stream onto a `cli` bar, and the
//! numbers a run produces do not depend on whether anyone was watching.

use std::sync::Mutex;

use ferx_tools::bootstrap::BootstrapEvent;
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// `{pos}/{len}` with a bar and an ETA. `{eta}` is indicatif's own estimate
/// from the completion rate so far, which is the right shape for this run: the
/// replicates are the same model on the same number of subjects, so their fit
/// times differ by convergence path rather than by size.
const BAR_TEMPLATE: &str = "  [{elapsed_precise}] [{bar:30}] {pos}/{len} {msg} (ETA {eta})";

/// The bars for one bootstrap run.
///
/// The whole state is behind one `Mutex` because the sink is called from
/// whichever Rayon worker finished a fit; the lock is taken once per completed
/// *fit*, so it contends with nothing.
pub struct BootstrapProgress {
    enabled: bool,
    state: Mutex<State>,
}

/// Which pass the bar on screen is counting.
///
/// The two bars are told apart by this rather than by their counts, because a
/// count cannot do it: `completed` is handed out by a `fetch_add` inside
/// `par_iter`, so the event carrying 1 is not necessarily the first one
/// delivered.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Pass {
    Replicates,
    DeltaOfv,
}

#[derive(Default)]
struct State {
    /// The base fit's spinner, live only while that fit runs.
    base: Option<ProgressBar>,
    /// The replicate bar, and then the Δofv bar. Only one is ever on screen:
    /// the Δofv pass starts after the last replicate is in.
    bar: Option<ProgressBar>,
    /// Which pass `bar` is counting, or `None` when there is no bar.
    pass: Option<Pass>,
    /// Replicates still to fit, as `Started` reported them - kept so the bar
    /// can be created when the base fit clears the screen rather than before.
    replicates: usize,
}

impl State {
    /// Put a fresh bar for `pass` on screen, clearing whatever was there.
    fn open(&mut self, pass: Pass, len: usize, unit: &str) {
        if let Some(previous) = self.bar.take() {
            previous.finish_and_clear();
        }
        self.bar = Some(bar(len, unit));
        self.pass = Some(pass);
    }

    /// Step the bar for `pass`, if that is the one on screen.
    ///
    /// Never backwards. `completed` is produced by a `fetch_add` inside
    /// `par_iter` and reported from whichever worker finished the fit, so 5 can
    /// reach us before 4 - and a bar that steps back also skews indicatif's
    /// `{eta}`, which it derives from the rate so far.
    fn advance(&self, pass: Pass, completed: usize) {
        if self.pass != Some(pass) {
            return;
        }
        if let Some(bar) = &self.bar {
            bar.set_position(bar.position().max(completed as u64));
        }
    }

    /// Take the bar and the spinner off screen.
    fn clear(&mut self) {
        if let Some(spinner) = self.base.take() {
            spinner.finish_and_clear();
        }
        if let Some(bar) = self.bar.take() {
            bar.finish_and_clear();
        }
        self.pass = None;
    }
}

impl BootstrapProgress {
    /// `enabled = false` makes every method a no-op - `--no-progress`, and any
    /// caller that would rather print its own thing.
    ///
    /// A non-terminal stderr needs no flag: `indicatif` resolves
    /// [`ProgressDrawTarget::stderr`] to a hidden target when stderr is
    /// redirected, so a bar in a log file or a pipe is already impossible.
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            state: Mutex::new(State::default()),
        }
    }

    /// Render one event. Never panics: it is called from inside `par_iter`,
    /// where a panic would take the whole run down with it.
    pub fn on_event(&self, event: BootstrapEvent) {
        if !self.enabled {
            return;
        }
        let mut state = match self.state.lock() {
            Ok(s) => s,
            // Poisoned means a previous render panicked. Losing the bar is the
            // correct response to that; losing the run is not.
            Err(_) => return,
        };
        match event {
            BootstrapEvent::Started {
                replicates,
                reused,
                base_fit,
            } => {
                if reused > 0 {
                    eprintln!("  reusing {reused} replicates already on disk");
                }
                state.replicates = replicates;
                if base_fit {
                    state.base = Some(spinner("fitting the base model"));
                } else {
                    state.open(Pass::Replicates, replicates, "replicates");
                }
            }
            BootstrapEvent::BaseFitDone => {
                if let Some(spinner) = state.base.take() {
                    spinner.finish_and_clear();
                }
                let replicates = state.replicates;
                state.open(Pass::Replicates, replicates, "replicates");
            }
            BootstrapEvent::ReplicateDone { completed, .. } => {
                state.advance(Pass::Replicates, completed);
            }
            BootstrapEvent::DeltaOfvDone { completed, total } => {
                // Any Δofv event is where the replicate bar is done: the pass
                // only starts once every replicate is in. Which one arrives
                // first is not known - the evaluations are handed their count by
                // a `fetch_add` inside `par_iter` and report from whichever
                // worker finished - so the swap keys on the pass rather than on
                // `completed == 1`, which would leave the first evaluations
                // drawn onto the replicate bar and then restart the Δofv bar
                // under them.
                if state.pass != Some(Pass::DeltaOfv) {
                    state.open(Pass::DeltaOfv, total, "delta-OFV evaluations");
                }
                state.advance(Pass::DeltaOfv, completed);
            }
            BootstrapEvent::Finished => state.clear(),
        }
    }
}

fn bar(len: usize, unit: &str) -> ProgressBar {
    let bar = ProgressBar::new(len as u64).with_message(unit.to_string());
    bar.set_draw_target(ProgressDrawTarget::stderr());
    // An unparseable template is a bug in the constant above, not a reason to
    // fail a bootstrap: fall back to indicatif's default style.
    if let Ok(style) = ProgressStyle::with_template(BAR_TEMPLATE) {
        bar.set_style(style.progress_chars("=> "));
    }
    bar
}

fn spinner(message: &str) -> ProgressBar {
    let spinner = ProgressBar::new_spinner().with_message(message.to_string());
    spinner.set_draw_target(ProgressDrawTarget::stderr());
    spinner.enable_steady_tick(std::time::Duration::from_millis(120));
    spinner
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Under `cargo test` stderr is not a terminal, so the bars resolve to a
    /// hidden draw target and nothing is written - but `position()` is still
    /// the state the bar would draw, which is what these assert.
    fn positions(progress: &BootstrapProgress) -> Option<u64> {
        progress
            .state
            .lock()
            .expect("not poisoned")
            .bar
            .as_ref()
            .map(|b| b.position())
    }

    #[test]
    fn the_replicate_bar_appears_after_the_base_fit_and_tracks_completions() {
        let progress = BootstrapProgress::new(true);
        progress.on_event(BootstrapEvent::Started {
            replicates: 3,
            reused: 0,
            base_fit: true,
        });
        // While the base model is fitting there is a spinner and no bar: a bar
        // sitting at 0/3 through a fit that can take minutes reads as stuck.
        assert!(positions(&progress).is_none());

        progress.on_event(BootstrapEvent::BaseFitDone);
        assert_eq!(positions(&progress), Some(0));

        progress.on_event(BootstrapEvent::ReplicateDone {
            completed: 2,
            total: 3,
        });
        assert_eq!(positions(&progress), Some(2));

        progress.on_event(BootstrapEvent::Finished);
        assert!(positions(&progress).is_none());
    }

    #[test]
    fn a_run_without_a_base_fit_starts_the_bar_immediately() {
        let progress = BootstrapProgress::new(true);
        progress.on_event(BootstrapEvent::Started {
            replicates: 5,
            reused: 2,
            base_fit: false,
        });
        assert_eq!(positions(&progress), Some(0));
        // The length is what is left to fit, not the sample count: the two
        // reused replicates are already on disk.
        let state = progress.state.lock().expect("not poisoned");
        assert_eq!(state.bar.as_ref().and_then(|b| b.length()), Some(5));
    }

    #[test]
    fn the_delta_ofv_pass_replaces_the_replicate_bar() {
        let progress = BootstrapProgress::new(true);
        progress.on_event(BootstrapEvent::Started {
            replicates: 2,
            reused: 0,
            base_fit: false,
        });
        progress.on_event(BootstrapEvent::ReplicateDone {
            completed: 2,
            total: 2,
        });
        progress.on_event(BootstrapEvent::DeltaOfvDone {
            completed: 1,
            total: 2,
        });
        // A fresh bar at 1, not the replicate bar carried on past its end.
        assert_eq!(positions(&progress), Some(1));

        progress.on_event(BootstrapEvent::DeltaOfvDone {
            completed: 2,
            total: 2,
        });
        assert_eq!(positions(&progress), Some(2));
    }

    #[test]
    fn the_delta_ofv_bar_appears_whichever_evaluation_reports_first() {
        // The evaluations take their `completed` from a `fetch_add` inside
        // `par_iter` and report from whichever worker finished, so the event
        // carrying 1 need not arrive first. Keying the swap on `completed == 1`
        // drew these onto the replicate bar and then restarted under them.
        // A resumed run, so the two passes have different lengths: 4 replicates
        // left to fit, but the Δofv sweep covers all 6 including the reused.
        let progress = BootstrapProgress::new(true);
        progress.on_event(BootstrapEvent::Started {
            replicates: 4,
            reused: 2,
            base_fit: false,
        });
        progress.on_event(BootstrapEvent::ReplicateDone {
            completed: 4,
            total: 4,
        });

        progress.on_event(BootstrapEvent::DeltaOfvDone {
            completed: 3,
            total: 6,
        });
        let state = progress.state.lock().expect("not poisoned");
        assert_eq!(state.pass, Some(Pass::DeltaOfv));
        // The Δofv bar, at 3 of 6 - not the replicate bar carried on past its
        // end, which would have shown 3 of 4 under the wrong label and then
        // been thrown away when `completed == 1` turned up.
        let bar = state.bar.as_ref().expect("a bar");
        assert_eq!(bar.position(), 3);
        assert_eq!(bar.length(), Some(6));
    }

    #[test]
    fn a_late_completion_does_not_step_the_bar_backwards() {
        let progress = BootstrapProgress::new(true);
        progress.on_event(BootstrapEvent::Started {
            replicates: 8,
            reused: 0,
            base_fit: false,
        });
        progress.on_event(BootstrapEvent::ReplicateDone {
            completed: 5,
            total: 8,
        });
        // Worker A finished fourth but reported after worker B finished fifth.
        progress.on_event(BootstrapEvent::ReplicateDone {
            completed: 4,
            total: 8,
        });
        assert_eq!(positions(&progress), Some(5));
    }

    #[test]
    fn a_disabled_progress_draws_nothing() {
        let progress = BootstrapProgress::new(false);
        progress.on_event(BootstrapEvent::Started {
            replicates: 4,
            reused: 0,
            base_fit: true,
        });
        progress.on_event(BootstrapEvent::BaseFitDone);
        progress.on_event(BootstrapEvent::ReplicateDone {
            completed: 1,
            total: 4,
        });
        assert!(positions(&progress).is_none());
    }
}
