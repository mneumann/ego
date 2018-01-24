// NOTE: Use std::ops::RangeInclusive once it becomes stable.
#[derive(Copy, Clone, Debug, Deserialize)]
pub struct RangeInclusive<T>
where
    T: PartialEq + PartialOrd,
{
    pub start: T,
    pub end: T,
}

impl<T: PartialEq + PartialOrd> RangeInclusive<T> {
    pub fn contains(&self, value: T) -> bool {
        value >= self.start && value <= self.end
    }
}
