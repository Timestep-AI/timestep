use crate::test_case::{TestCase, TestResult, TestStatus};
use anyhow::{Context, Result};
use super::python_bridge::PythonBridge;
use std::path::PathBuf;
use super::typescript_bridge::TypeScriptBridge;
use tracing::{info, warn};
use serde::Serialize;

pub struct TestHarness {
    python_bridge: Option<PythonBridge>,
    typescript_bridge: Option<TypeScriptBridge>,
}

impl TestHarness {
    pub fn new() -> Result<Self> {
        let python_bridge = PythonBridge::new().ok();
        let typescript_bridge = TypeScriptBridge::new().ok();

        if python_bridge.is_none() {
            warn!("Python bridge not available");
        }
        if typescript_bridge.is_none() {
            warn!("TypeScript bridge not available");
        }

        Ok(Self {
            python_bridge,
            typescript_bridge,
        })
    }

    pub async fn run_tests(
        &self,
        test_path: PathBuf,
        run_python: bool,
        run_typescript: bool,
        filter: Option<&str>,
    ) -> Result<TestResults> {
        let test_cases = self.load_test_cases(test_path, filter)?;
        let mut all_results = Vec::new();

        info!("Running {} test cases", test_cases.len());

        for test_case in test_cases {
            info!("Running test: {}", test_case.name);

            if run_python {
                if let Some(ref bridge) = self.python_bridge {
                    let result = bridge.execute_test(&test_case).await?;
                    all_results.push(result);
                }
            }

            if run_typescript {
                if let Some(ref bridge) = self.typescript_bridge {
                    let result = bridge.execute_test(&test_case).await?;
                    all_results.push(result);
                }
            }
        }

        let total = all_results.len();
        let passed = all_results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Passed))
            .count();
        let failed = all_results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Failed))
            .count();
        let errors = all_results
            .iter()
            .filter(|r| matches!(r.status, TestStatus::Error))
            .count();

        Ok(TestResults {
            results: all_results,
            total,
            passed,
            failed,
            errors,
        })
    }

    fn load_test_cases(
        &self,
        test_path: PathBuf,
        filter: Option<&str>,
    ) -> Result<Vec<TestCase>> {
        let mut test_cases = Vec::new();

        if test_path.is_file() {
            let content = std::fs::read_to_string(&test_path)
                .with_context(|| format!("Failed to read test file: {:?}", test_path))?;
            
            // For now, support JSON. TOML support can be added later if needed
            let ext = test_path.extension()
                .and_then(|s| s.to_str())
                .unwrap_or("json");
            
            match ext {
                "json" => {
                    // Parse as Value first to handle arrays in model_name and provider_config
                    let test_case_value: serde_json::Value = serde_json::from_str(&content)
                        .with_context(|| format!("Failed to parse JSON test case: {:?}", test_path))?;
                    
                    // Expand parameterized test cases (e.g., arrays in provider_config and model_name)
                    let expanded_values = self.expand_parameterized_test_value(test_case_value)?;
                    for case_value in expanded_values {
                        let test_case: TestCase = serde_json::from_value(case_value)
                            .with_context(|| format!("Failed to deserialize expanded test case: {:?}", test_path))?;
                        if filter.map(|f| test_case.name.contains(f)).unwrap_or(true) {
                            test_cases.push(test_case);
                        }
                    }
                }
                "toml" => {
                    // TOML support - parse array of tests
                    let tests: toml::Value = toml::from_str(&content)
                        .with_context(|| format!("Failed to parse TOML: {:?}", test_path))?;
                    
                    if let Some(tests_array) = tests.get("test").and_then(|t| t.as_array()) {
                        for test_val in tests_array {
                            let test_str = toml::to_string(test_val)?;
                            // Convert TOML to JSON for parsing
                            let json_str = serde_json::to_string(&toml::from_str::<toml::Value>(&test_str)?)?;
                            let test_case_value: serde_json::Value = serde_json::from_str(&json_str)?;
                            
                            // Expand parameterized test cases
                            let expanded_values = self.expand_parameterized_test_value(test_case_value)?;
                            for case_value in expanded_values {
                                let test_case: TestCase = serde_json::from_value(case_value)?;
                                if filter.map(|f| test_case.name.contains(f)).unwrap_or(true) {
                                    test_cases.push(test_case);
                                }
                            }
                        }
                    }
                }
                _ => {
                    anyhow::bail!("Unsupported test file format: {}", ext);
                }
            }
        } else if test_path.is_dir() {
            // Load all test files from directory
            for entry in std::fs::read_dir(&test_path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_file() {
                    let ext = path.extension()
                        .and_then(|s| s.to_str())
                        .unwrap_or("");
                    if ext == "json" || ext == "toml" {
                        let mut cases = self.load_test_cases(path, filter)?;
                        test_cases.append(&mut cases);
                    }
                }
            }
        }

        Ok(test_cases)
    }

    /// Expand a test case with parameterized values (arrays) into multiple test cases
    /// Works with JSON Value to handle arrays in both provider_config and input fields
    fn expand_parameterized_test_value(&self, test_case_value: serde_json::Value) -> Result<Vec<serde_json::Value>> {
        use serde_json::Value;
        
        // Find parameterized fields in provider_config
        let mut provider_params: Vec<(String, Vec<Value>)> = Vec::new();
        if let Some(setup) = test_case_value.get("setup") {
            if let Some(provider_config) = setup.get("provider_config") {
                if let Value::Object(config_obj) = provider_config {
                    for (key, value) in config_obj {
                        if let Value::Array(arr) = value {
                            if !arr.is_empty() {
                                provider_params.push((format!("provider_config.{}", key), arr.clone()));
                            }
                        }
                    }
                }
            }
        }
        
        // Find parameterized fields in input (e.g., model_name)
        let mut input_params: Vec<(String, Vec<Value>)> = Vec::new();
        if let Some(input) = test_case_value.get("input") {
            if let Some(model_name_val) = input.get("model_name") {
                if let Value::Array(arr) = model_name_val {
                    if !arr.is_empty() {
                        input_params.push(("input.model_name".to_string(), arr.clone()));
                    }
                }
            }
        }
        
        // Combine all parameterized fields
        let mut all_params = provider_params;
        all_params.extend(input_params);
        
        // If no parameterized fields, return the test case as-is
        if all_params.is_empty() {
            return Ok(vec![test_case_value]);
        }
        
        // Generate cartesian product of all parameter combinations
        let mut expanded_cases = Vec::new();
        self.generate_combinations_value(test_case_value, &all_params, 0, &mut Vec::new(), &mut expanded_cases)?;
        
        Ok(expanded_cases)
    }
    
    /// Recursively generate all combinations of parameterized values
    fn generate_combinations_value(
        &self,
        base_case: serde_json::Value,
        params: &[(String, Vec<serde_json::Value>)],
        param_idx: usize,
        current_values: &mut Vec<(String, serde_json::Value)>,
        results: &mut Vec<serde_json::Value>,
    ) -> Result<()> {
        use serde_json::Value;
        
        if param_idx >= params.len() {
            // All parameters assigned, create a test case
            let mut new_case = base_case.clone();
            let mut name_suffixes = Vec::new();
            
            // Get the base name
            let base_name = new_case.get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("test")
                .to_string();
            
            for (param_path, param_value) in current_values.iter() {
                // Update the appropriate field based on param_path
                if param_path.starts_with("provider_config.") {
                    let key = param_path.strip_prefix("provider_config.").unwrap();
                    if let Some(setup) = new_case.get_mut("setup") {
                        if let Some(provider_config) = setup.get_mut("provider_config") {
                            if let Value::Object(config_obj) = provider_config {
                                config_obj.insert(key.to_string(), param_value.clone());
                                name_suffixes.push(format!("{}={}", key, self.format_value(param_value)));
                            }
                        }
                    }
                } else if param_path == "input.model_name" {
                    if let Some(input) = new_case.get_mut("input") {
                        if let Value::Object(input_obj) = input {
                            input_obj.insert("model_name".to_string(), param_value.clone());
                            if let Value::String(s) = param_value {
                                name_suffixes.push(format!("model={}", s));
                            }
                        }
                    }
                }
            }
            
            // Update the name with all parameter values
            if let Some(name) = new_case.get_mut("name") {
                let mut new_name = base_name.clone();
                if !name_suffixes.is_empty() {
                    new_name.push_str(&format!(" ({})", name_suffixes.join(", ")));
                }
                *name = Value::String(new_name);
            }
            
            results.push(new_case);
            return Ok(());
        }
        
        // Try each value for the current parameter
        let (param_path, param_values) = &params[param_idx];
        for param_value in param_values {
            current_values.push((param_path.clone(), param_value.clone()));
            self.generate_combinations_value(base_case.clone(), params, param_idx + 1, current_values, results)?;
            current_values.pop();
        }
        
        Ok(())
    }
    
    fn format_value(&self, value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::String(s) => s.clone(),
            _ => "variant".to_string(),
        }
    }
}

#[derive(Debug, Serialize)]
pub struct TestResults {
    pub results: Vec<TestResult>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub errors: usize,
}

impl TestResults {
    pub fn print_text_report(&self) {
        println!("\n=== Test Results ===");
        println!("Total: {}", self.total);
        println!("Passed: {}", self.passed);
        println!("Failed: {}", self.failed);
        println!("Errors: {}", self.errors);
        println!("\n=== Detailed Results ===\n");

        for result in &self.results {
            let status_str = match result.status {
                TestStatus::Passed => "✓ PASSED",
                TestStatus::Failed => "✗ FAILED",
                TestStatus::Error => "⚠ ERROR",
                TestStatus::Skipped => "⊘ SKIPPED",
            };
            println!("{} [{}] {}", status_str, result.implementation, result.test_name);
            if let Some(ref error) = result.error {
                println!("  Error: {}", error);
            }
            if !result.assertion_failures.is_empty() {
                for failure in &result.assertion_failures {
                    println!("  Assertion failed: {}", failure.reason);
                }
            }
        }
    }

    pub fn to_html_report(&self) -> Result<String> {
        // Simple HTML report generation
        let mut html = String::from(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Timestep Test Results</title>
    <style>
        body { font-family: monospace; margin: 20px; }
        .passed { color: green; }
        .failed { color: red; }
        .error { color: orange; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Test Results</h1>
    <p>Total: {} | Passed: {} | Failed: {} | Errors: {}</p>
    <table>
        <tr>
            <th>Status</th>
            <th>Implementation</th>
            <th>Test Name</th>
            <th>Duration (ms)</th>
            <th>Error</th>
        </tr>
"#,
        );
        html = html.replace("{}", &self.total.to_string());
        html = html.replace("{}", &self.passed.to_string());
        html = html.replace("{}", &self.failed.to_string());
        html = html.replace("{}", &self.errors.to_string());

        for result in &self.results {
            let status_class = match result.status {
                TestStatus::Passed => "passed",
                TestStatus::Failed => "failed",
                TestStatus::Error => "error",
                TestStatus::Skipped => "error",
            };
            let status_str = format!("{:?}", result.status);
            let error = result.error.as_deref().unwrap_or("");
            html.push_str(&format!(
                r#"<tr class="{}">
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                    <td>{}</td>
                </tr>"#,
                status_class, status_str, result.implementation, result.test_name, result.duration_ms, error
            ));
        }

        html.push_str("</table>\n</body>\n</html>");
        Ok(html)
    }
}

