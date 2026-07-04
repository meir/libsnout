use std::marker::PhantomData;


pub trait Shape: Copy + Into<usize> + From<usize> {
    fn count() -> usize;

    fn iter() -> impl Iterator<Item = Self> {
        (0..Self::count()).map(Self::from)
    }
}

pub struct Weights<S, T=f32> {
    data: Vec<Option<T>>,
    _phantom: PhantomData<fn(S)>,
}

impl<S: Shape, T> Weights<S, T> {
    pub fn new() -> Self {
        let mut data = Vec::new();
        data.resize_with(S::count(), || None);

        Self {
            data,
            _phantom: PhantomData,
        }
    }

    pub fn clear(&mut self) {
        for i in 0..S::count() {
            self.data[i] = None;
        }
    }

    pub fn set(&mut self, index: S, value: T) {
        let index = index.into();
        self.data[index] = Some(value);
    }

    pub fn get(&self, index: S) -> Option<T> where T: Copy {
        let index = index.into();
        self.data[index]
    }

    pub fn fill_with(&mut self, outputs: &[T], map: &[Option<S>]) where T: Copy {
        self.clear();

        for (i, &shape) in map.iter().enumerate() {
            if let Some(shape) = shape {
                let index = shape.into();
                self.data[index] = Some(outputs[i]);
            }
        }
    }

    pub fn iter(&self) -> impl Iterator<Item = (S, T)> + '_ where T: Copy {
        S::iter().filter_map(|s| {
            let index = s.into();
            self.data[index].map(|v| (s, v))
        })
    }
}

impl<S: Shape, T> Default for Weights<S, T> {
    fn default() -> Self {
        Self::new()
    }
}
