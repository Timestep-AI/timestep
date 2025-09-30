import * as readline from "readline"
import { TimestepAgent } from "./ag_ui_server"
import { randomUUID } from "node:crypto"

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
})

async function chatLoop() {
  console.log("🤖 Timestep Assistant started!")
  console.log("Type your messages and press Enter. Press Ctrl+D to quit.\n")

  const modelId = process.env.MODEL_ID ?? 'ollama/gpt-oss:120b-cloud'
  const openaiUseResponses = (process.env.OPENAI_USE_RESPONSES ?? 'false').toLowerCase() === 'true'

  const agent = new TimestepAgent(modelId, openaiUseResponses)

  const messages: any[] = []
  const threadId = randomUUID()

  return new Promise<void>((resolve) => {
    const promptUser = () => {
      rl.question("> ", async (input) => {
        if (input.trim() === "") {
          promptUser()
          return
        }
        console.log("")

        // Pause input while processing
        rl.pause()

        // Add user message to conversation
        const userMessage = {
          id: randomUUID(),
          role: "user" as const,
          content: input.trim(),
        }
        messages.push(userMessage)

        try {
          // Create run input for AG-UI
          const runInput = {
            threadId,
            runId: randomUUID(),
            messages,
          }

          // Subscribe to the agent's Observable stream
          const stream$ = agent['run'](runInput)

          let hasStartedMessage = false

          stream$.subscribe({
            next: (event: any) => {
              if (event.type === 'TEXT_MESSAGE_START') {
                if (!hasStartedMessage) {
                  process.stdout.write("🤖 Assistant: ")
                  hasStartedMessage = true
                }
              } else if (event.type === 'TEXT_MESSAGE_CHUNK') {
                process.stdout.write(event.delta)
              } else if (event.type === 'TEXT_MESSAGE_END') {
                // Message complete
              } else if (event.type === 'TOOL_CALL_START') {
                process.stdout.write(`\n🔧 Calling tool: ${event.toolName}\n`)
              } else if (event.type === 'TOOL_CALL_ARGS') {
                process.stdout.write(`   Arguments: ${JSON.stringify(event.args)}\n`)
              } else if (event.type === 'TOOL_CALL_RESULT') {
                process.stdout.write(`   Result: ${event.result}\n`)
              } else if (event.type === 'TOOL_CALL_END') {
                process.stdout.write(`✅ Tool call complete\n\n`)
              }
            },
            error: (error) => {
              console.error("❌ Error:", error)
              rl.resume()
              promptUser()
            },
            complete: () => {
              if (hasStartedMessage) {
                console.log("\n")
              }
              rl.resume()
              promptUser()
            }
          })
        } catch (error) {
          console.error("❌ Error:", error)
          rl.resume()
          promptUser()
        }
      })
    }

    // Handle Ctrl+D to quit
    rl.on("close", () => {
      console.log("\n👋 Thanks for using AG-UI Assistant!")
      resolve()
    })

    promptUser()
  })
}

async function main() {
  await chatLoop()
}

main().catch(console.error)
