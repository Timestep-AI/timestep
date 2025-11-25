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
        Commands::Bump { python, typescript } => {
            bump_versions(python, typescript).await?;
        }
    }
    
    Ok(())
}

async fn bump_versions(bump_python: bool, bump_typescript: bool) -> Result<()> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .context("Failed to find workspace root")?;
    
    if bump_python {
        println!("Bumping Python version...");
        let python_dir = workspace_root.join("python");
        
        // Check if uv is available
        let uv_output = Command::new("uv")
            .arg("--version")
            .output()
            .await;
        
        if uv_output.is_ok() {
            let output = Command::new("uv")
                .current_dir(&python_dir)
                .arg("version")
                .arg("patch")
                .output()
                .await
                .context("Failed to run uv version patch")?;
            
            if !output.status.success() {
                let stderr = String::from_utf8_lossy(&output.stderr);
                eprintln!("Warning: Failed to bump Python version: {}", stderr);
            } else {
                let stdout = String::from_utf8_lossy(&output.stdout);
                println!("Python version bumped: {}", stdout.trim());
            }
        } else {
            eprintln!("Warning: uv not found, skipping Python version bump");
        }
    }
    
    if bump_typescript {
        println!("Bumping TypeScript version...");
        let typescript_dir = workspace_root.join("typescript");
        
        // Use npm version patch (works with npx too)
        let output = Command::new("npm")
            .current_dir(&typescript_dir)
            .arg("version")
            .arg("patch")
            .arg("--no-git-tag-version")
            .arg("--no-commit-hooks")
            .output()
            .await
            .context("Failed to run npm version patch")?;
        
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            eprintln!("Warning: Failed to bump TypeScript version: {}", stderr);
        } else {
            let stdout = String::from_utf8_lossy(&output.stdout);
            println!("TypeScript version bumped: {}", stdout.trim());
        }
    }
    
    Ok(())
}

