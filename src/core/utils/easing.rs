//! 渐变函数

use std::f64::consts::PI;

use pyo3::pyclass;
use strum_macros::{Display, EnumString};

/// Easing functions
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum Easing {
    /// Linear
    #[strum(to_string = "Linear")]
    Linear,
    /// Sine In
    #[strum(to_string = "Sine In")]
    SineIn,
    /// Sine Out
    #[strum(to_string = "Sine Out")]
    SineOut,
    /// Sine In/Out
    #[strum(to_string = "Sine In/Out")]
    SineInOut,
    /// Quart In
    #[strum(to_string = "Quart In")]
    QuartIn,
    /// Quart Out
    #[strum(to_string = "Quart Out")]
    QuartOut,
    /// Quart In/Out
    #[strum(to_string = "Quart In/Out")]
    QuartInOut,
    /// Cubic In
    #[strum(to_string = "Cubic In")]
    CubicIn,
    /// Cubic Out
    #[strum(to_string = "Cubic Out")]
    CubicOut,
    /// Cubic In/Out
    #[strum(to_string = "Cubic In/Out")]
    CubicInOut,
    /// Circ In
    #[strum(to_string = "Circ In")]
    CircIn,
    /// Circ Out
    #[strum(to_string = "Circ Out")]
    CircOut,
    /// Circ In/Out
    #[strum(to_string = "Circ In/Out")]
    CircInOut,
    /// Back In
    #[strum(to_string = "Back In")]
    BackIn,
    /// Back Out
    #[strum(to_string = "Back Out")]
    BackOut,
    /// Back In/Out
    #[strum(to_string = "Back In/Out")]
    BackInOut,
    /// Elastic In
    #[strum(to_string = "Elastic In")]
    ElasticIn,
    /// Elastic Out
    #[strum(to_string = "Elastic Out")]
    ElasticOut,
    /// Elastic In/Out
    #[strum(to_string = "Elastic In/Out")]
    ElasticInOut,
    /// Bounce In
    #[strum(to_string = "Bounce In")]
    BounceIn,
    /// Bounce Out
    #[strum(to_string = "Bounce Out")]
    BounceOut,
    /// Bounce In/Out
    #[strum(to_string = "Bounce In/Out")]
    BounceInOut,
}

impl Easing {
    /// 应用缓动函数
    pub fn apply(&self, t: f64) -> f64 {
        match self {
            Easing::Linear => Self::ease_linear(t),
            Easing::BackIn => Self::ease_in_back(t),
            Easing::BackOut => Self::ease_out_back(t),
            Easing::BackInOut => Self::ease_in_out_back(t),
            Easing::ElasticIn => Self::ease_in_elastic(t),
            Easing::ElasticOut => Self::ease_out_elastic(t),
            Easing::ElasticInOut => Self::ease_in_out_elastic(t),
            Easing::BounceIn => Self::ease_in_bounce(t),
            Easing::BounceOut => Self::ease_out_bounce(t),
            Easing::BounceInOut => Self::ease_in_out_bounce(t),
            Easing::QuartIn => Self::ease_in_quart(t),
            Easing::QuartOut => Self::ease_out_quart(t),
            Easing::QuartInOut => Self::ease_in_out_quart(t),
            Easing::CubicIn => Self::ease_in_cubic(t),
            Easing::CubicOut => Self::ease_out_cubic(t),
            Easing::CubicInOut => Self::ease_in_out_cubic(t),
            Easing::CircIn => Self::ease_in_circ(t),
            Easing::CircOut => Self::ease_out_circ(t),
            Easing::CircInOut => Self::ease_in_out_circ(t),
            Easing::SineIn => Self::ease_in_sine(t),
            Easing::SineOut => Self::ease_out_sine(t),
            Easing::SineInOut => Self::ease_in_out_sine(t),
        }
    }
}

impl Easing {
    fn ease_linear(t: f64) -> f64 {
        t
    }

    /// Back 缓动函数
    fn ease_in_back(t: f64) -> f64 {
        let s = 1.70158;
        t * t * ((s + 1.0) * t - s)
    }

    fn ease_out_back(t: f64) -> f64 {
        let s = 1.70158;
        let t = t - 1.0;
        t * t * ((s + 1.0) * t + s) + 1.0
    }

    fn ease_in_out_back(t: f64) -> f64 {
        let s = 1.70158 * 1.525;
        if t < 0.5 {
            let t = t * 2.0;
            (t * t * (t * (s + 1.0) - s)) / 2.0
        } else {
            let t = t * 2.0 - 2.0;
            (t * t * ((s + 1.0) * t + s) + 2.0) / 2.0
        }
    }

    /// Elastic 缓动函数
    fn ease_in_elastic(t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }

        let p = 0.3;
        let s = p / 4.0;
        let t = t - 1.0;

        -(2.0f64.powf(10.0 * t) * ((t - s) * (2.0 * PI) / p).sin())
    }

    fn ease_out_elastic(t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }

        let p = 0.3;
        let s = p / 4.0;
        2.0f64.powf(-10.0 * t) * ((t - s) * (2.0 * PI) / p).sin() + 1.0
    }

    fn ease_in_out_elastic(t: f64) -> f64 {
        if t <= 0.0 {
            return 0.0;
        }
        if t >= 1.0 {
            return 1.0;
        }

        let p = 0.3 * 1.5;
        let s = p / 4.0;
        let t = t * 2.0 - 1.0;

        if t < 0.0 {
            -0.5 * (2.0f64.powf(10.0 * t) * ((t - s) * (2.0 * PI) / p).sin())
        } else {
            0.5 * 2.0f64.powf(-10.0 * t) * ((t - s) * (2.0 * PI) / p).sin() + 1.0
        }
    }

    /// Bounce 缓动函数
    fn ease_in_bounce(t: f64) -> f64 {
        1.0 - Self::ease_out_bounce(1.0 - t)
    }

    fn ease_out_bounce(t: f64) -> f64 {
        if t < 1.0 / 2.75 {
            7.5625 * t * t
        } else if t < 2.0 / 2.75 {
            let t = t - 1.5 / 2.75;
            7.5625 * t * t + 0.75
        } else if t < 2.5 / 2.75 {
            let t = t - 2.25 / 2.75;
            7.5625 * t * t + 0.9375
        } else {
            let t = t - 2.625 / 2.75;
            7.5625 * t * t + 0.984375
        }
    }

    fn ease_in_out_bounce(t: f64) -> f64 {
        if t < 0.5 {
            Self::ease_in_bounce(t * 2.0) * 0.5
        } else {
            Self::ease_out_bounce(t * 2.0 - 1.0) * 0.5 + 0.5
        }
    }

    /// Quart 缓动函数
    fn ease_in_quart(t: f64) -> f64 {
        t * t * t * t
    }

    fn ease_out_quart(t: f64) -> f64 {
        let t = t - 1.0;
        1.0 - t * t * t * t
    }

    fn ease_in_out_quart(t: f64) -> f64 {
        if t < 0.5 {
            8.0 * t * t * t * t
        } else {
            let t = t - 1.0;
            1.0 - 8.0 * t * t * t * t
        }
    }

    /// Cubic 缓动函数
    fn ease_in_cubic(t: f64) -> f64 {
        t * t * t
    }

    fn ease_out_cubic(t: f64) -> f64 {
        let t = t - 1.0;
        t * t * t + 1.0
    }

    fn ease_in_out_cubic(t: f64) -> f64 {
        if t < 0.5 {
            4.0 * t * t * t
        } else {
            let t = t - 1.0;
            4.0 * t * t * t + 1.0
        }
    }

    /// Circ 缓动函数
    fn ease_in_circ(t: f64) -> f64 {
        1.0 - (1.0 - t * t).sqrt()
    }

    fn ease_out_circ(t: f64) -> f64 {
        let t = t - 1.0;
        (1.0 - t * t).sqrt()
    }

    fn ease_in_out_circ(t: f64) -> f64 {
        if t < 0.5 {
            (1.0 - (1.0 - (2.0 * t).powi(2)).sqrt()) / 2.0
        } else {
            let t = 2.0 * t - 2.0;
            (1.0 - (1.0 - t * t).sqrt()) / 2.0 + 0.5
        }
    }

    /// Sine 缓动函数
    fn ease_in_sine(t: f64) -> f64 {
        1.0 - (t * PI / 2.0).cos()
    }

    fn ease_out_sine(t: f64) -> f64 {
        (t * PI / 2.0).sin()
    }

    fn ease_in_out_sine(t: f64) -> f64 {
        -((PI * t).cos() - 1.0) / 2.0
    }
}
