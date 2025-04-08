//! 类型定义

use std::fmt;

/// 节点类型
#[derive(Debug)]
pub enum NodeType {
    STRING,
    BOOLEAN,
    LIST,
    ANY,
}

impl fmt::Display for NodeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeType::STRING => write!(f, "STRING"),
            NodeType::BOOLEAN => write!(f, "BOOLEAN"),
            NodeType::LIST => write!(f, "LIST"),
            NodeType::ANY => write!(f, "ANY"),
        }
    }
}
