//! 渐变函数

use std::f64::consts::PI;

use pyo3::pyclass;
use strum_macros::{Display, EnumString};

use crate::error::Error;

/// Easing functions
#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, EnumString, Display)]
#[strum(serialize_all = "kebab-case")]
pub enum Easing {
    /// Linear
    /// 线性缓动
    #[strum(to_string = "Linear")]
    Linear,
    /// Sine In
    /// 缓入正弦曲线
    #[strum(to_string = "Sine In")]
    SineIn,
    /// Sine Out
    /// 缓出正弦曲线
    #[strum(to_string = "Sine Out")]
    SineOut,
    /// Sine In/Out
    /// 缓入缓出正弦曲线
    #[strum(to_string = "Sine In/Out")]
    SineInOut,
    /// Quart In
    /// 缓入四次方曲线
    #[strum(to_string = "Quart In")]
    QuartIn,
    /// Quart Out
    /// 缓出四次方曲线
    #[strum(to_string = "Quart Out")]
    QuartOut,
    /// Quart In/Out
    /// 缓入缓出四次方曲线
    #[strum(to_string = "Quart In/Out")]
    QuartInOut,
    /// Cubic In
    ///  缓入三次方曲线
    #[strum(to_string = "Cubic In")]
    CubicIn,
    /// Cubic Out
    /// 缓出三次方曲线
    #[strum(to_string = "Cubic Out")]
    CubicOut,
    /// Cubic In/Out
    /// 缓入缓出三次方曲线
    #[strum(to_string = "Cubic In/Out")]
    CubicInOut,
    /// Circ In
    /// 缓入圆形曲线
    #[strum(to_string = "Circ In")]
    CircIn,
    /// Circ Out
    /// 缓出圆形曲线
    #[strum(to_string = "Circ Out")]
    CircOut,
    /// Circ In/Out
    /// 缓入缓出圆形曲线
    #[strum(to_string = "Circ In/Out")]
    CircInOut,
    /// Back In
    /// 缓入回弹效果
    #[strum(to_string = "Back In")]
    BackIn,
    /// Back Out
    /// 缓出回弹效果
    #[strum(to_string = "Back Out")]
    BackOut,
    /// Back In/Out
    /// 缓入缓出回弹效果
    #[strum(to_string = "Back In/Out")]
    BackInOut,
    /// Elastic In
    /// 缓入弹性效果
    #[strum(to_string = "Elastic In")]
    ElasticIn,
    /// Elastic Out
    /// 缓出弹性效果
    #[strum(to_string = "Elastic Out")]
    ElasticOut,
    /// Elastic In/Out
    /// 缓入缓出弹性效果
    #[strum(to_string = "Elastic In/Out")]
    ElasticInOut,
    /// Bounce In
    /// 缓入弹跳效果
    #[strum(to_string = "Bounce In")]
    BounceIn,
    /// Bounce Out
    /// 缓出弹跳效果
    #[strum(to_string = "Bounce Out")]
    BounceOut,
    /// Bounce In/Out
    /// 缓入缓出弹跳效果
    #[strum(to_string = "Bounce In/Out")]
    BounceInOut,
}

impl Easing {
    /// 应用缓动函数到指定进度值
    ///
    /// # 参数
    /// - `t`: 动画进度值，范围 [0.0, 1.0]
    ///
    /// # 返回值
    /// 应用缓动函数后的进度值
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
    /// 线性缓动函数
    fn ease_linear(t: f64) -> f64 {
        t
    }

    /// Back 缓动函数
    ///
    /// 缓入回弹效果
    ///
    /// 在动画开始时产生回弹效果，适用于需要强调开始的动作
    fn ease_in_back(t: f64) -> f64 {
        let s = 1.70158;
        t * t * ((s + 1.0) * t - s)
    }

    /// 缓出回弹效果
    ///
    /// 在动画结束时产生回弹效果，适用于需要强调结束的动作
    fn ease_out_back(t: f64) -> f64 {
        let s = 1.70158;
        let t = t - 1.0;
        t * t * ((s + 1.0) * t + s) + 1.0
    }

    /// 缓入缓出回弹效果
    ///
    /// 在动画开始和结束时都产生回弹效果
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
    ///
    /// 缓入弹性效果
    ///
    /// 模拟弹簧被拉回后释放的效果，开始时回弹幅度较大
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

    /// 缓出弹性效果
    ///
    /// 模拟弹簧释放后的振荡效果，结束时回弹幅度较大
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

    /// 缓入缓出弹性效果
    ///
    /// 在动画开始和结束时都产生弹性振荡效果
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
    ///
    /// 缓入弹跳效果
    ///
    /// 模拟物体落地弹跳效果，开始时弹跳幅度较大
    fn ease_in_bounce(t: f64) -> f64 {
        1.0 - Self::ease_out_bounce(1.0 - t)
    }

    /// 缓出弹跳效果
    ///
    /// 模拟物体落地弹跳效果，结束时弹跳幅度较大
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

    /// 缓入缓出弹跳效果
    ///
    /// 在动画开始和结束时都产生弹跳效果
    fn ease_in_out_bounce(t: f64) -> f64 {
        if t < 0.5 {
            Self::ease_in_bounce(t * 2.0) * 0.5
        } else {
            Self::ease_out_bounce(t * 2.0 - 1.0) * 0.5 + 0.5
        }
    }

    /// Quart 缓动函数
    ///
    /// 缓入四次方曲线
    ///
    /// 基于t⁴的缓入效果，起始变化较平缓，后期加速
    fn ease_in_quart(t: f64) -> f64 {
        t * t * t * t
    }

    /// 缓出四次方曲线
    ///
    /// 基于t⁴的缓出效果，起始变化较快，后期逐渐平缓
    fn ease_out_quart(t: f64) -> f64 {
        let t = t - 1.0;
        1.0 - t * t * t * t
    }

    /// 缓入缓出四次方曲线
    ///
    /// 结合缓入和缓出的四次方曲线效果，提供平滑的过渡
    fn ease_in_out_quart(t: f64) -> f64 {
        if t < 0.5 {
            8.0 * t * t * t * t
        } else {
            let t = t - 1.0;
            1.0 - 8.0 * t * t * t * t
        }
    }

    /// Cubic 缓动函数
    ///
    /// 提供基于三次方曲线的缓动效果
    ///
    /// 基于t³的缓入效果，起始变化较平缓，后期加速
    fn ease_in_cubic(t: f64) -> f64 {
        t * t * t
    }

    /// 缓出三次方曲线
    ///
    /// 基于t³的缓出效果，起始变化较快，后期逐渐平缓
    fn ease_out_cubic(t: f64) -> f64 {
        let t = t - 1.0;
        t * t * t + 1.0
    }

    /// 缓入缓出三次方曲线
    ///
    /// 结合缓入和缓出的三次方曲线效果，提供平滑的过渡
    fn ease_in_out_cubic(t: f64) -> f64 {
        if t < 0.5 {
            4.0 * t * t * t
        } else {
            let t = t - 1.0;
            4.0 * t * t * t + 1.0
        }
    }

    /// Circ 缓动函数
    ///
    /// 提供基于圆形曲线的缓动效果
    ///
    /// 模拟圆形路径的缓入效果，起始变化较慢，后期加速
    fn ease_in_circ(t: f64) -> f64 {
        1.0 - (1.0 - t * t).sqrt()
    }

    /// 缓出圆形曲线
    ///
    /// 模拟圆形路径的缓出效果，起始变化较快，后期逐渐平缓
    fn ease_out_circ(t: f64) -> f64 {
        let t = t - 1.0;
        (1.0 - t * t).sqrt()
    }

    /// 缓入缓出圆形曲线
    ///
    /// 结合缓入和缓出的圆形曲线效果，提供平滑的过渡
    fn ease_in_out_circ(t: f64) -> f64 {
        if t < 0.5 {
            (1.0 - (1.0 - (2.0 * t).powi(2)).sqrt()) / 2.0
        } else {
            let t = 2.0 * t - 2.0;
            (1.0 - (1.0 - t * t).sqrt()) / 2.0 + 0.5
        }
    }

    /// Sine 缓动函数
    ///
    /// 提供基于正弦曲线的缓动效果
    ///
    /// 基于正弦函数的缓入效果，起始变化较慢，后期加速
    fn ease_in_sine(t: f64) -> f64 {
        1.0 - (t * PI / 2.0).cos()
    }

    /// 缓出正弦曲线
    ///
    /// 基于正弦函数的缓出效果，起始变化较快，后期逐渐平缓
    fn ease_out_sine(t: f64) -> f64 {
        (t * PI / 2.0).sin()
    }

    /// 缓入缓出正弦曲线
    ///
    /// 结合缓入和缓出的正弦曲线效果，提供平滑的过渡
    fn ease_in_out_sine(t: f64) -> f64 {
        -((PI * t).cos() - 1.0) / 2.0
    }
}

impl TryFrom<String> for Easing {
    type Error = Error;

    fn try_from(easing: String) -> Result<Self, Self::Error> {
        easing
            .parse::<Easing>()
            .map_err(|e| Error::ParseEnumString(e.to_string()))
    }
}
