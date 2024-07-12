# Signal Hound Spike Spectrum Analyzer SHR Parser

This Rust library provides powerful tools for parsing and handling SHR files generated by the Signal Hound Spike softwa.
The SHR file format encompasses a file header and multiple sweeps, each with its own header and data.
This library leverages memory mapping for efficient file reading and Rayon for parallel processing of sweeps, ensuring
high performance and scalability.

## Features

- **Parse SHR Files:** Read and interpret SHR files, including headers and sweeps.
- **Validate Files:** Ensure the integrity of SHR files by validating signatures and versions.
- **Sweep Metrics Calculation:** Compute key metrics such as peak, mean, and low values from sweep data.
- **Serialization & Deserialization:** Utilize Serde for seamless serialization and deserialization of SHR data.
- **CSV Export:** Export parsed SHR data to CSV format for easy analysis and reporting.

## Installation

To include this library in your project, add the following dependencies to your `Cargo.toml` file:

```toml
[dependencies]
shr_parser = "1.0.5"
```

## Usage

Here's an example of how to use the SHR file parser:

```rust
use std::path::PathBuf;
use shr_parser::{SHRParser, SHRParsingType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let file_path = PathBuf::from("path/to/your/shrfile.shr");
    let parser = SHRParser::new(file_path, SHRParsingType::Peak)?;

    println!("{}", parser.to_str());

    parser.to_csv(PathBuf::from("output.csv"))?;

    Ok(())
}
```

## Documentation

Complete documentation is available on [docs.rs](https://docs.rs/shr_parser/). To generate the
documentation locally, run:

```sh
cargo doc --open
```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open an issue or
submit a pull request.

## License

This project is licensed under the GPL-3 License. See the `LICENSE` file for details.
