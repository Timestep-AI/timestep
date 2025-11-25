use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod harness;
mod test_case;

use harness::TestHarness;

#[derive(Parser)]
#[command(name = "timestep-test")]
#[command(about = "Behavior test harness for Timestep implementations")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run behavior tests
    Test {
        /// Test case file or directory
        #[arg(short, long, default_value = "tests/behavior")]
        test_path: PathBuf,
        
        /// Run tests against Python implementation
        #[arg(long, default_value_t = true)]
        python: bool,
        
        /// Run tests against TypeScript implementation
        #[arg(long, default_value_t = true)]
        typescript: bool,
        
        /// Filter tests by name pattern
        #[arg(short, long)]
        filter: Option<String>,
        
        /// Output format: json, text, html
        #[arg(long, default_value = "text")]
        format: String,
        
        /// Output file for report
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Test {
            test_path,
            python,
            typescript,
            filter,
            format,
            output,
        } => {
            let harness = TestHarness::new()?;
            
            let results = harness
                .run_tests(test_path, python, typescript, filter.as_deref())
                .await?;
            
            match format.as_str() {
                "json" => {
                    let json = serde_json::to_string_pretty(&results)?;
                    if let Some(output_path) = output {
                        std::fs::write(output_path, json)?;
                    } else {
                        println!("{}", json);
                    }
                }
                "html" => {
                    let html = results.to_html_report()?;
                    if let Some(output_path) = output {
                        std::fs::write(output_path, html)?;
                    } else {
                        println!("{}", html);
                    }
                }
                _ => {
                    results.print_text_report();
                }
            }
        }
    }
    
    Ok(())
}

