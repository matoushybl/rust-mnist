use crate::image::{Image, IMAGE_SIZE};
use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{ErrorKind, Read};

pub fn load_dataset(image_path: &str, label_path: &str) -> Result<Vec<Image>, std::io::Error> {
    let mut images_file = File::open(image_path)?;
    let mut labels_file = File::open(label_path)?;

    let _ = images_file.read_u32::<BigEndian>()?;
    let images_count = images_file.read_u32::<BigEndian>()?;
    let _ = images_file.read_u32::<BigEndian>()?;
    let _ = images_file.read_u32::<BigEndian>()?;

    let _ = labels_file.read_u32::<BigEndian>()?;
    let labels_count = labels_file.read_u32::<BigEndian>()?;

    if images_count != labels_count {
        return Err(std::io::Error::new(
            ErrorKind::Other,
            "Dataset length mismatch",
        ));
    }

    let mut labels: Vec<u8> = Vec::new();
    let mut pixels: Vec<[u8; IMAGE_SIZE]> = Vec::new();
    for _ in 0..labels_count {
        labels.push(labels_file.read_u8()?);
        let mut pxs = [0u8; IMAGE_SIZE];
        images_file.read_exact(&mut pxs)?;
        pixels.push(pxs);
    }

    let images: Vec<Image> = labels
        .iter()
        .zip(pixels)
        .map(|(label, pixels)| Image {
            label: *label,
            pixels,
        })
        .collect();

    Ok(images)
}
