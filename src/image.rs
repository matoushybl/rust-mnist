use std::fmt::{Display, Formatter};

pub const IMAGE_SIZE: usize = 28 * 28;

pub struct Image {
    pub(crate) label: u8,
    pub(crate) pixels: [u8; IMAGE_SIZE],
}

impl Display for Image {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "label: {}\n", self.label)?;
        let chunks = self.pixels.chunks(28);
        for chunk in chunks {
            for px in chunk {
                if *px > 128 {
                    write!(f, "  ")?;
                } else {
                    write!(f, "██")?;
                }
            }

            write!(f, "\n")?;
        }

        write!(f, "\n\n")
    }
}
