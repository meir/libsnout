use std::collections::VecDeque;

pub struct JsonFramer {
    buf: String,
    depth: i32,
    in_string: bool,
    escape: bool,
    ready: VecDeque<String>,
}

impl JsonFramer {
    pub fn new() -> Self {
        Self {
            buf: String::new(),
            depth: 0,
            in_string: false,
            escape: false,
            ready: VecDeque::new(),
        }
    }

    pub fn feed(&mut self, data: &str) {
        for ch in data.chars() {
            self.buf.push(ch);

            if self.escape {
                self.escape = false;
                continue;
            }

            if self.in_string {
                match ch {
                    '\\' => self.escape = true,
                    '"' => self.in_string = false,
                    _ => {}
                }
                continue;
            }

            match ch {
                '"' => self.in_string = true,
                '{' => self.depth += 1,
                '}' => {
                    self.depth -= 1;
                    if self.depth == 0 {
                        self.ready.push_back(std::mem::take(&mut self.buf));
                    }
                }
                _ => {}
            }
        }
    }

    pub fn next(&mut self) -> Option<String> {
        self.ready.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_message() {
        let mut framer = JsonFramer::new();
        framer.feed(r#"{"key": "value"}"#);
        assert_eq!(framer.next().as_deref(), Some(r#"{"key": "value"}"#));
        assert_eq!(framer.next(), None);
    }

    #[test]
    fn two_messages_one_chunk() {
        let mut framer = JsonFramer::new();
        framer.feed(r#"{"a": 1}{"b": 2}"#);
        assert_eq!(framer.next().as_deref(), Some(r#"{"a": 1}"#));
        assert_eq!(framer.next().as_deref(), Some(r#"{"b": 2}"#));
        assert_eq!(framer.next(), None);
    }

    #[test]
    fn split_across_chunks() {
        let mut framer = JsonFramer::new();
        framer.feed(r#"{"ke"#);
        assert_eq!(framer.next(), None);
        framer.feed(r#"y": "val"#);
        assert_eq!(framer.next(), None);
        framer.feed(r#"ue"}"#);
        assert!(framer.next().is_some());
        assert_eq!(framer.next(), None);
    }

    #[test]
    fn braces_inside_strings() {
        let mut framer = JsonFramer::new();
        framer.feed(r#"{"data": "{}}{{"}"#);
        assert_eq!(framer.next().as_deref(), Some(r#"{"data": "{}}{{"}"#));
    }

    #[test]
    fn escaped_quotes() {
        let mut framer = JsonFramer::new();
        framer.feed(r#"{"msg": "say \"hi\""}"#);
        assert!(framer.next().is_some());
        assert_eq!(framer.next(), None);
    }

    #[test]
    fn leading_whitespace_ignored() {
        let mut framer = JsonFramer::new();
        framer.feed("  \n  {\"a\": 1}");
        assert!(framer.next().is_some());
        assert_eq!(framer.next(), None);
    }
}
