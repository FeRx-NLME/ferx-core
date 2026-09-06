//! The seam between a search's step logic and the [`Runner`]: one step's
//! candidates in, one [`RunReport`] out.
//!
//! Every stepwise tool of the #1175 epic — covsearch (#1180), modelsearch
//! (#1181) — makes its decisions by comparing numbers the fitter returns, and
//! none of those decisions is observable through real fits without putting
//! every test in the slow tier. So the tools are written over this trait and
//! tested against a scripted implementation, while production uses
//! [`RunnerFitter`]: a [`Runner`] per step, journalled under
//! `<dir>/<step_dir>` when there is a directory. One implementation shared
//! rather than one per tool, so a change to how a step is journalled or
//! cancelled cannot land in one tool and not the other.

use std::path::PathBuf;

use ferx_core::{CancelFlag, Population};

use super::{Candidate, RunOptions, RunReport, Runner};

/// Fits one step's candidates.
pub(crate) trait StepFitter {
    /// `step_dir` is the directory name a cached run would journal this step
    /// under (`base`, `forward-1`, `layer-2`, …).
    fn fit_step(&self, step_dir: &str, candidates: &[Candidate]) -> Result<RunReport, String>;
}

/// The production fitter: a [`Runner`] per step.
pub(crate) struct RunnerFitter<'a> {
    pub threads: usize,
    pub dir: Option<PathBuf>,
    pub cancel: Option<CancelFlag>,
    pub data: &'a Population,
    pub options: RunOptions,
}

impl StepFitter for RunnerFitter<'_> {
    fn fit_step(&self, step_dir: &str, candidates: &[Candidate]) -> Result<RunReport, String> {
        let mut runner = Runner::new().threads(self.threads);
        if let Some(dir) = &self.dir {
            runner = runner.cache_dir(dir.join(step_dir));
        }
        if let Some(flag) = &self.cancel {
            runner = runner.cancel(flag.clone());
        }
        runner.run(candidates, self.data, &self.options)
    }
}
