use crate::test_case::{TestCase, TestResult, TestStatus};
use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::process::Command;
use tracing::debug;
use which::which;

pub struct PythonBridge {
    python_path: PathBuf,
    test_runner_path: PathBuf,
}

impl PythonBridge {
    pub fn new() -> Result<Self> {
        // Try uv first, then python3/python
        let python_path = if which("uv").is_ok() {
            which("uv").unwrap()
        } else {
            which("python3")
                .or_else(|_| which("python"))
                .context("Python not found in PATH. Install Python or uv")?
        };

        // Path to Python test runner script
        let test_runner_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("python")
            .join("tests")
            .join("run_test.py");

        if !test_runner_path.exists() {
            anyhow::bail!("Python test runner not found at {:?}", test_runner_path);
        }

        Ok(Self {
            python_path,
            test_runner_path,
        })
    }

    pub async fn execute_test(&self, test_case: &TestCase) -> Result<TestResult> {
        let start = std::time::Instant::now();
        
        // Serialize test case to JSON
        let test_json = serde_json::to_string(test_case)?;
        
        debug!("Executing Python test: {}", test_case.name);
        
        // Write test case to temp file for Python script to read
        let temp_file = tempfile::NamedTempFile::new()?;
        std::fs::write(temp_file.path(), test_json)?;
        
        // Get the python directory (parent of tests/)
        let python_dir = self.test_runner_path.parent().unwrap().parent().unwrap();
        
        // Use uv run if python_path is uv, otherwise use python directly
        let output = if self.python_path.file_name().and_then(|n| n.to_str()) == Some("uv") {
            Command::new(&self.python_path)
                .current_dir(python_dir)
                .arg("run")
                .arg("python")
                .arg(&self.test_runner_path)
                .arg(temp_file.path())
                .output()
                .await
        } else {
            Command::new(&self.python_path)
                .current_dir(python_dir)
                .arg(&self.test_runner_path)
                .arg(temp_file.path())
                .output()
                .await
        }
        .context("Failed to execute Python test")?;

        let duration_ms = start.elapsed().as_millis() as u64;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Ok(TestResult {
                test_name: test_case.name.clone(),
                implementation: "python".to_string(),
                status: TestStatus::Error,
                duration_ms,
                error: Some(format!("Python execution failed: {}", stderr)),
                actual_result: None,
                assertion_failures: vec![],
            });
        }

        // Parse result from stdout
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: TestResult = serde_json::from_str(&stdout)
            .context("Failed to parse Python test result")?;

        Ok(result)
    }
}

