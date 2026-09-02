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

#[derive(Default)]
struct State {
    /// The base fit's spinner, live only while that fit runs.
    base: Option<ProgressBar>,
    /// The replicate bar, and then the Δofv bar. Only one is ever on screen:
    /// the Δofv pass starts after the last replicate is in.
    bar: Option<ProgressBar>,
    /// Replicates still to fit, as `Started` reported them - kept so the bar
    /// can be created when the base fit clears the screen rather than before.
    replicates: usize,
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
                    state.bar = Some(bar(replicates, "replicates"));
                }
            }
            BootstrapEvent::BaseFitDone => {
                if let Some(spinner) = state.base.take() {
                    spinner.finish_and_clear();
                }
                let replicates = state.replicates;
                state.bar = Some(bar(replicates, "replicates"));
            }
            BootstrapEvent::ReplicateDone { completed, .. } => {
                if let Some(bar) = &state.bar {
                    bar.set_position(completed as u64);
                }
            }
            BootstrapEvent::DeltaOfvDone { completed, total } => {
                // The first Δofv event is where the replicate bar is done: the
                // pass only starts once every replicate is in.
                if completed == 1 || state.bar.is_none() {
                    if let Some(previous) = state.bar.take() {
                        previous.finish_and_clear();
                    }
                    state.bar = Some(bar(total, "delta-OFV evaluations"));
                }
                if let Some(bar) = &state.bar {
                    bar.set_position(completed as u64);
                }
            }
            BootstrapEvent::Finished => {
                if let Some(spinner) = state.base.take() {
                    spinner.finish_and_clear();
                }
                if let Some(bar) = state.bar.take() {
                    bar.finish_and_clear();
                }
            }
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
