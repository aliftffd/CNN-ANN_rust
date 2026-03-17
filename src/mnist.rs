use std::fs::File;
use std::io::Read;

pub struct MnistData {
    pub train_images: Vec<Vec<f64>>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<Vec<f64>>,
    pub test_labels: Vec<u8>,
}

fn read_u32(file: &mut File) -> u32 {
    let mut buf = [0u8; 4];
    file.read_exact(&mut buf).unwrap();
    u32::from_be_bytes(buf)
}

fn load_images(path: &str) -> Vec<Vec<f64>> {
    let mut file = File::open(path).expect(&format!("Cannot open {}", path));

    let magic = read_u32(&mut file);
    assert_eq!(magic, 2051, "Not an MNIST image file");

    let num_images = read_u32(&mut file) as usize;
    let rows = read_u32(&mut file) as usize;
    let cols = read_u32(&mut file) as usize;
    let pixels = rows * cols;

    let mut images = Vec::with_capacity(num_images);
    let mut buf = vec![0u8; pixels];

    for _ in 0..num_images {
        file.read_exact(&mut buf).unwrap();
        let image: Vec<f64> = buf.iter().map(|&b| b as f64 / 255.0).collect();
        images.push(image);
    }

    images
}

fn load_labels(path: &str) -> Vec<u8> {
    let mut file = File::open(path).expect(&format!("Cannot open {}", path));

    let magic = read_u32(&mut file);
    assert_eq!(magic, 2049, "Not an MNIST label file");

    let num_labels = read_u32(&mut file) as usize;

    let mut labels = vec![0u8; num_labels];
    file.read_exact(&mut labels).unwrap();

    labels
}

pub fn load_mnist(data_dir: &str) -> MnistData {
    println!("Loading MNIST from {}...", data_dir);

    let train_images = load_images(&format!("{}/train-images-idx3-ubyte", data_dir));
    let train_labels = load_labels(&format!("{}/train-labels-idx1-ubyte", data_dir));
    let test_images = load_images(&format!("{}/t10k-images-idx3-ubyte", data_dir));
    let test_labels = load_labels(&format!("{}/t10k-labels-idx1-ubyte", data_dir));

    println!(
        "Loaded {} train, {} test images",
        train_images.len(),
        test_images.len()
    );

    MnistData {
        train_images,
        train_labels,
        test_images,
        test_labels,
    }
}
