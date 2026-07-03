use std::{borrow::Cow, cell::Cell, rc::Rc, time::{Duration, Instant}};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use console::style;

pub trait StatusBarItem {
    fn render(&self) -> Cow<'static, str>;
}

pub struct StatusBar {
    spinner: ProgressBar,
    items: Vec<Rc<dyn StatusBarItem>>,
}

impl StatusBar {
    pub fn new(multi: &MultiProgress) -> Self {
        let spinner = multi.add(ProgressBar::new_spinner());

        spinner.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠁⠂⠄⡀⡈⡐⡠⣀⣁⣂⣄⣌⣔⣤⣥⣦⣮⣶⣷⣿⡿⠿⢟⠟⡛⠛⠫⢋⠋⠍⡉⠉⠑⠡⢁"),
        );

        Self { spinner, items: Vec::new() }
    }

    /// Register an item, rendered in insertion order. The returned handle can
    /// be used to update the item's state from elsewhere (it shares ownership
    /// with the bar).
    pub fn add<T: StatusBarItem + 'static>(&mut self, item: T) -> Rc<T> {
        let item = Rc::new(item);
        self.items.push(item.clone());
        item
    }

    pub fn display(&mut self) {
        let items = self.items.iter()
            .map(|item| item.render())
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        self.spinner.set_message(items);
        self.spinner.tick();
    }
}

/// A freshness indicator: shows `label` in green while it has been "beaten"
/// within `timeout`, and red once it goes stale. Starts stale until the first
/// [`beat`](Heartbeat::beat), so it reads red until real data arrives.
pub struct Heartbeat {
    label: Cow<'static, str>,
    timeout: Duration,
    last_beat: Cell<Option<Instant>>,
}

impl Heartbeat {
    pub fn new(label: impl Into<Cow<'static, str>>, timeout: Duration) -> Self {
        Self {
            label: label.into(),
            timeout,
            last_beat: Cell::new(None),
        }
    }

    /// Mark the signal as freshly received.
    pub fn beat(&self) {
        self.last_beat.set(Some(Instant::now()));
    }

    fn is_alive(&self) -> bool {
        self.last_beat
            .get()
            .is_some_and(|last| last.elapsed() <= self.timeout)
    }
}

impl StatusBarItem for Heartbeat {
    fn render(&self) -> Cow<'static, str> {
        let label = style(self.label.as_ref());
        let styled = if self.is_alive() {
            label.green()
        } else {
            label.red()
        };
        styled.to_string().into()
    }
}

/// How often [`Rate`] recomputes the value it displays.
const SAMPLE_INTERVAL: Duration = Duration::from_secs(1);

/// A throughput counter: shows `label:<rate>/s`, where the rate is the number
/// of [`inc`](Rate::inc) calls per second. The displayed value is resampled at
/// most once per [`SAMPLE_INTERVAL`], so it holds steady between updates rather
/// than flickering on every frame.
pub struct Rate {
    label: Cow<'static, str>,
    precision: usize,
    count: Cell<u32>,
    window_start: Cell<Instant>,
    last_rate: Cell<f32>,
}

impl Rate {
    pub fn new(label: impl Into<Cow<'static, str>>, precision: usize) -> Self {
        Self {
            label: label.into(),
            precision,
            count: Cell::new(0),
            window_start: Cell::new(Instant::now()),
            last_rate: Cell::new(0.0),
        }
    }

    /// Record a single event.
    pub fn inc(&self) {
        self.count.set(self.count.get() + 1);
    }

    /// Once the sampling window has elapsed, convert the accumulated count into
    /// a per-second rate and start a fresh window.
    fn resample(&self) {
        let elapsed = self.window_start.get().elapsed();
        if elapsed >= SAMPLE_INTERVAL {
            self.last_rate
                .set(self.count.get() as f32 / elapsed.as_secs_f32());
            self.count.set(0);
            self.window_start.set(Instant::now());
        }
    }
}

impl StatusBarItem for Rate {
    fn render(&self) -> Cow<'static, str> {
        self.resample();
        format!(
            "{}:{:.*}/s",
            self.label,
            self.precision,
            self.last_rate.get()
        )
        .into()
    }
}

pub struct Pair {
    label: Cow<'static, str>,
    left: Cell<f32>,
    right: Cell<f32>,
}

impl Pair {
    pub fn new(label: impl Into<Cow<'static, str>>) -> Self {
        Self {
            label: label.into(),
            left: Cell::new(0.0),
            right: Cell::new(0.0),
        }
    }

    pub fn set(&self, left: f32, right: f32) {
        self.left.set(left);
        self.right.set(right);
    }
}

impl StatusBarItem for Pair {
    fn render(&self) -> Cow<'static, str> {
        format!(
            "{}:{:.0}|{:.0}",
            self.label,
            self.left.get() * 100.0,
            self.right.get() * 100.0
        )
        .into()
    }
}
