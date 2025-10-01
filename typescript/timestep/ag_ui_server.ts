import {
  AbstractAgent,
  RunAgentInput,
  EventType,
  BaseEvent,
} from "@ag-ui/client"
import { Observable } from "rxjs"
import { z } from 'zod'
import { Agent, Runner, tool, OpenAIProvider, RunToolApprovalItem } from '@openai/agents'
import { RunConfig, RunState } from '@openai/agents-core'
import { MultiProvider, MultiProviderMap } from './multi_provider'
import { OllamaModelProvider } from './ollama_model_provider'
import { fetchMcpTools } from './mcp_server_proxy'
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs'

export class TimestepAgent extends AbstractAgent {
  private modelId: string
  private openaiUseResponses: boolean

  constructor(modelId?: string, openaiUseResponses: boolean = false) {
    super()
    this.modelId = modelId ?? process.env.MODEL_ID ?? 'ollama/gpt-oss:120b-cloud'
    this.openaiUseResponses = openaiUseResponses
  }

  protected run(input: RunAgentInput): Observable<BaseEvent> {
    return new Observable<BaseEvent>((observer) => {
      observer.next({
        type: EventType.RUN_STARTED,
        threadId: input.threadId,
        runId: input.runId,
      } as any)

      this.runAgent(input, observer)
        .then(() => {
          observer.next({
            type: EventType.RUN_FINISHED,
            threadId: input.threadId,
            runId: input.runId,
          } as any)
          observer.complete()
        })
        .catch((error) => {
          observer.next({
            type: EventType.RUN_ERROR,
            message: error.message,
          } as any)
          observer.error(error)
        })
    })
  }

  private async runAgent(input: RunAgentInput, observer: any) {
    // Set up model providers
    const modelProviderMap = new MultiProviderMap()

    modelProviderMap.addProvider("ollama", new OllamaModelProvider({
      apiKey: process.env.OLLAMA_API_KEY,
    }))

    modelProviderMap.addProvider("anthropic", new OpenAIProvider({
      apiKey: process.env.ANTHROPIC_API_KEY,
      baseURL: "https://api.anthropic.com/v1/",
      useResponses: false
    }))

    const modelProvider = new MultiProvider({
      provider_map: modelProviderMap,
      openai_use_responses: this.openaiUseResponses,
    })

    const runConfig: RunConfig = {
      modelProvider: modelProvider,
      traceIncludeSensitiveData: true,
      tracingDisabled: false,
    }

    // Configure approval policies (get_weather now requires approval for AG-UI testing)
    const requireApproval = {
      never: { toolNames: ['search_codex_code', 'fetch_codex_documentation'] },
      always: { toolNames: ['get_weather', 'fetch_generic_url_content'] },
    }

    // Fetch built-in tools for weather agent
    console.log('[MCP] Loading built-in tools...')
    const weatherTools = await fetchMcpTools(null, true, requireApproval)
    console.log(`[MCP] Loaded ${weatherTools.length} built-in tools`)

    // Create weather agent with built-in tools
    const weatherAgent = new Agent({
      model: this.modelId,
      name: 'Weather agent',
      instructions: 'You provide weather information.',
      handoffDescription: 'Handles weather-related queries',
      tools: weatherTools,
    })

    // Fetch remote MCP tools from the codex server
    console.log('[MCP] Fetching tools from https://gitmcp.io/timestep-ai/timestep...')
    const mcpTools = await fetchMcpTools('https://gitmcp.io/timestep-ai/timestep', false, requireApproval)
    console.log(`[MCP] Loaded ${mcpTools.length} remote tools`)

    // Create main agent with remote MCP tools and weather handoff
    const agent = new Agent({
      model: this.modelId,
      name: 'Main Assistant',
      instructions:
        'You are a helpful assistant. For questions about the timestep-ai/timestep repository, use the MCP tools. For weather questions, hand off to the weather agent.',
      tools: mcpTools,
      handoffs: [weatherAgent],
    })

    const runner = new Runner(runConfig)

    // Ensure data directory exists for state persistence
    mkdirSync('data', { recursive: true })

    // Check if this is a resume from an interrupt
    let agentInput: string | RunState<undefined, Agent<unknown, "text">>
    let isResumingFromInterrupt = false

    if (input.resume?.interruptId) {
      // Load saved state from disk
      const stateFilename = `data/${input.threadId}.state.json`
      if (existsSync(stateFilename)) {
        console.log('[AG-UI] Resuming from interrupt...')
        const raw = readFileSync(stateFilename, 'utf8')
        const savedState = JSON.parse(raw)
        const runState = await RunState.fromString(agent, JSON.stringify(savedState))

        // Find the tool approval item
        const approvalItem = runState._generatedItems
          .filter(item => item.type === 'tool_approval_item')
          .find(item => (item as RunToolApprovalItem).rawItem?.id === input.resume?.interruptId) as RunToolApprovalItem | undefined

        if (approvalItem) {
          // Apply the approval decision from resume payload
          const approved = input.resume.payload?.approved ?? false
          const alwaysApprove = input.resume.payload?.alwaysApprove ?? false
          const alwaysReject = input.resume.payload?.alwaysReject ?? false

          if (approved) {
            runState.approve(approvalItem, { alwaysApprove })
            console.log('[AG-UI] Tool approved')
          } else {
            runState.reject(approvalItem, { alwaysReject })
            console.log('[AG-UI] Tool rejected')
          }

          agentInput = runState as RunState<undefined, Agent<unknown, "text">>
          isResumingFromInterrupt = true
        } else {
          console.error('[AG-UI] Warning: Could not find approval item for interrupt ID')
          // Fall back to normal execution
          const messages = input.messages ?? []
          const userMessages = messages.filter(m => m.role === 'user')
          const lastUserMessage = userMessages[userMessages.length - 1]
          agentInput = lastUserMessage?.content ?? ''
        }
      } else {
        console.error('[AG-UI] Warning: No saved state found for resume')
        // Fall back to normal execution
        const messages = input.messages ?? []
        const userMessages = messages.filter(m => m.role === 'user')
        const lastUserMessage = userMessages[userMessages.length - 1]
        agentInput = lastUserMessage?.content ?? ''
      }
    } else {
      // Normal execution - get the last user message as input
      const messages = input.messages ?? []
      const userMessages = messages.filter(m => m.role === 'user')
      const lastUserMessage = userMessages[userMessages.length - 1]
      agentInput = lastUserMessage?.content ?? ''

      if (!agentInput) {
        // No user input, just return without doing anything
        return
      }
    }

    const stream = await runner.run(
      agent,
      agentInput,
      { stream: true },
    )

    let currentMessageId: string | null = null
    let messageStarted = false

    for await (const chunk of stream) {
      if ('name' in chunk) {
        // Handle tool approval requests - save state and emit interrupt
        if (chunk.name === 'tool_approval_requested') {
          const toolApprovalItem = (chunk as any).item
          if (toolApprovalItem?.rawItem?.providerData?.function?.name) {
            const toolName = toolApprovalItem.rawItem.providerData.function.name
            const toolArguments = toolApprovalItem.rawItem.providerData.function.arguments
            const toolAgent = toolApprovalItem.agent?.name
            const interruptId = toolApprovalItem.rawItem.id || Date.now().toString()

            console.log('[AG-UI] Tool approval required, saving state...')

            // Save state to disk
            const state = (stream as any).state as RunState<any, any>
            const stateFilename = `data/${input.threadId}.state.json`
            try {
              const savedState = JSON.parse(JSON.stringify(state))
              writeFileSync(stateFilename, JSON.stringify(savedState, null, 2))
              console.log(`[AG-UI] State saved to ${stateFilename}`)
            } catch (err) {
              console.error('[AG-UI] Failed to serialize stream state:', err)
              throw err
            }

            // Emit RUN_FINISHED with interrupt outcome
            observer.next({
              type: EventType.RUN_FINISHED,
              threadId: input.threadId,
              runId: input.runId,
              outcome: 'interrupt',
              interrupt: {
                id: interruptId,
                reason: 'tool_approval_required',
                payload: {
                  toolName,
                  toolArguments,
                  agentName: toolAgent,
                }
              }
            } as any)

            // Complete the stream
            return
          }
        }

        // Check for message output events that contain text
        else if (chunk.name === 'message_output_created') {
          const messageItem = (chunk as any).item

          // Check if this is a message with text content
          if (messageItem?.rawItem?.content) {
            const content = messageItem.rawItem.content

            // Initialize message ID and send TEXT_MESSAGE_START
            if (!currentMessageId) {
              currentMessageId = messageItem.rawItem.id || Date.now().toString()
              observer.next({
                type: EventType.TEXT_MESSAGE_START,
                messageId: currentMessageId,
              } as any)
              messageStarted = true
            }

            // Handle array of content blocks
            if (Array.isArray(content)) {
              for (const part of content) {
                if ((part.type === 'text' || part.type === 'output_text') && part.text) {
                  observer.next({
                    type: EventType.TEXT_MESSAGE_CHUNK,
                    messageId: currentMessageId,
                    delta: part.text,
                  } as any)
                }
              }
            }
            // Handle string content
            else if (typeof content === 'string') {
              observer.next({
                type: EventType.TEXT_MESSAGE_CHUNK,
                messageId: currentMessageId,
                delta: content,
              } as any)
            }
          }
        }

        // Handle tool call events
        else if (chunk.name === 'tool_called') {
          const toolCallItem = (chunk as any).item
          if (toolCallItem?.rawItem?.providerData?.function) {
            const toolCallId = toolCallItem.rawItem.id || Date.now().toString()
            const toolName = toolCallItem.rawItem.providerData.function.name
            const toolArgs = toolCallItem.rawItem.providerData.function.arguments

            observer.next({
              type: EventType.TOOL_CALL_START,
              toolCallId,
              toolName,
            } as any)

            observer.next({
              type: EventType.TOOL_CALL_ARGS,
              toolCallId,
              args: toolArgs,
            } as any)
          }
        }

        else if (chunk.name === 'tool_output') {
          const toolOutputItem = (chunk as any).item
          if (toolOutputItem?.rawItem?.name) {
            const toolCallId = toolOutputItem.rawItem.tool_call_id || Date.now().toString()
            const result = toolOutputItem.output

            observer.next({
              type: EventType.TOOL_CALL_RESULT,
              toolCallId,
              result,
            } as any)

            observer.next({
              type: EventType.TOOL_CALL_END,
              toolCallId,
            } as any)
          }
        }
      }
    }

    // Send TEXT_MESSAGE_END if we started a message
    if (messageStarted && currentMessageId) {
      observer.next({
        type: EventType.TEXT_MESSAGE_END,
        messageId: currentMessageId,
      } as any)
    }

    await stream.completed
  }
}
