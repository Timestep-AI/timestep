use crate::test_case::{TestCase, TestResult, TestStatus};
use anyhow::{Context, Result};
use std::path::PathBuf;
use tokio::process::Command;
use tracing::debug;
use which::which;

pub struct TypeScriptBridge {
    node_path: PathBuf,
    test_runner_path: PathBuf,
}

impl TypeScriptBridge {
    pub fn new() -> Result<Self> {
        // Try node, then nodejs
        let node_path = which("node")
            .or_else(|_| which("nodejs"))
            .context("Node.js not found in PATH")?;

        // Path to TypeScript test runner script
        let test_runner_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .unwrap()
            .join("typescript")
            .join("tests")
            .join("run_test.ts");

        if !test_runner_path.exists() {
            anyhow::bail!("TypeScript test runner not found at {:?}", test_runner_path);
        }

        Ok(Self {
            node_path,
            test_runner_path,
        })
    }

    pub async fn execute_test(&self, test_case: &TestCase) -> Result<TestResult> {
        let start = std::time::Instant::now();
        
        // Serialize test case to JSON
        let test_json = serde_json::to_string(test_case)?;
        
        debug!("Executing TypeScript test: {}", test_case.name);
        
        // Write test case to temp file
        let temp_file = tempfile::NamedTempFile::new()?;
        std::fs::write(temp_file.path(), test_json)?;
        
        // Get the typescript directory (parent of tests/)
        let typescript_dir = self.test_runner_path.parent().unwrap().parent().unwrap();
        
        // Try to execute TypeScript test runner
        // Use npx tsx (doesn't require global installation)
        let output = if which("npx").is_ok() {
            // Try npx tsx first
            Command::new("npx")
                .current_dir(typescript_dir)
                .arg("--yes")
                .arg("tsx")
                .arg(&self.test_runner_path)
                .arg(temp_file.path())
                .output()
                .await
        } else if which("tsx").is_ok() {
            // Fallback: try tsx if globally installed
            Command::new("tsx")
                .current_dir(typescript_dir)
                .arg(&self.test_runner_path)
                .arg(temp_file.path())
                .output()
                .await
        } else {
            // Last resort: try with node directly (will fail for .ts files)
            Command::new(&self.node_path)
                .current_dir(typescript_dir)
                .arg(&self.test_runner_path)
                .arg(temp_file.path())
                .output()
                .await
        }
        .context("Failed to execute TypeScript test. Install tsx: npx tsx or npm install -g tsx")?;

        let duration_ms = start.elapsed().as_millis() as u64;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Ok(TestResult {
                test_name: test_case.name.clone(),
                implementation: "typescript".to_string(),
                status: TestStatus::Error,
                duration_ms,
                error: Some(format!("TypeScript execution failed: {}", stderr)),
                actual_result: None,
                assertion_failures: vec![],
            });
        }

        // Parse result from stdout
        let stdout = String::from_utf8_lossy(&output.stdout);
        let result: TestResult = serde_json::from_str(&stdout)
            .context("Failed to parse TypeScript test result")?;

        Ok(result)
    }
}

