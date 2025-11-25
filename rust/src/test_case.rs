use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestCase {
    pub name: String,
    pub description: Option<String>,
    pub setup: TestSetup,
    pub input: TestInput,
    pub expected: TestExpected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestSetup {
    pub provider_type: String,
    #[serde(default)]
    pub provider_config: HashMap<String, serde_json::Value>,
    #[serde(default)]
    pub mock_responses: Vec<MockResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MockResponse {
    pub endpoint: String,
    pub method: String,
    pub response: serde_json::Value,
    pub status_code: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestInput {
    pub model_name: String,
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub options: HashMap<String, serde_json::Value>,
    /// If true, create and run an Agent instead of just testing provider
    #[serde(default)]
    pub run_agent: bool,
    /// If true, use streaming mode for agent execution
    #[serde(default)]
    pub stream: bool,
    /// Agent configuration (system prompt, tools, etc.)
    #[serde(default)]
    pub agent_config: Option<AgentConfig>,
    /// User input for the agent (can be string or structured input)
    #[serde(default)]
    pub user_input: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    #[serde(default)]
    pub system_prompt: Option<String>,
    #[serde(default)]
    pub tools: Vec<Tool>,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    pub name: String,
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExpected {
    pub provider_type: Option<String>,
    pub model_name: Option<String>,
    pub should_succeed: Option<bool>,
    pub error_type: Option<String>,
    pub response_contains: Option<Vec<String>>,
    /// Expected agent output (for agent tests)
    pub agent_output: Option<AgentOutputExpectation>,
    #[serde(default)]
    pub assertions: Vec<Assertion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutputExpectation {
    /// Expected text content in the response
    #[serde(default)]
    pub contains_text: Option<Vec<String>>,
    /// Exact match (rarely used, as LLM outputs vary)
    #[serde(default)]
    pub exact_match: Option<String>,
    /// Minimum response length
    #[serde(default)]
    pub min_length: Option<usize>,
    /// Maximum response length
    #[serde(default)]
    pub max_length: Option<usize>,
    /// Response should match this regex pattern
    #[serde(default)]
    pub matches_pattern: Option<String>,
    /// Response should not contain these strings
    #[serde(default)]
    pub excludes_text: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Assertion {
    pub field: String,
    pub operator: String, // "equals", "contains", "matches", etc.
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    pub test_name: String,
    pub implementation: String,
    pub status: TestStatus,
    pub duration_ms: u64,
    pub error: Option<String>,
    pub actual_result: Option<serde_json::Value>,
    pub assertion_failures: Vec<AssertionFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "PascalCase")]
pub enum TestStatus {
    Passed,
    Failed,
    Error,
    Skipped,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssertionFailure {
    pub assertion: Assertion,
    pub actual_value: serde_json::Value,
    pub reason: String,
}

