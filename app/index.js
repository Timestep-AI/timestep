class ChatApp {
    constructor() {
        this.messages = [];
        this.currentTaskId = null;
        this.currentContextId = null;
        this.currentAgentId = null;
        this.eventSource = null;
        this.isProcessing = false;
        this.contexts = [];
        this.agents = [];
        this.agentCard = null;
        
        this.initializeElements();
        this.attachEventListeners();
        this.updateStatus('ready', 'Ready');
        this.loadAgents();
        this.loadAllContexts();
    }
    
    initializeElements() {
        this.messagesContainer = document.getElementById('messages');
        this.inputForm = document.getElementById('inputForm');
        this.messageInput = document.getElementById('messageInput');
        this.sendButton = document.getElementById('sendButton');
        this.statusIndicator = document.getElementById('statusIndicator');
        this.statusText = document.getElementById('statusText');
        this.contextsList = document.getElementById('contextsList');
        this.newChatButton = document.getElementById('newChatButton');
        this.agentSelect = document.getElementById('agentSelect');
        this.agentCardContainer = document.getElementById('agentCardContainer');
        this.copyChatButton = document.getElementById('copyChatButton');
    }
    
    attachEventListeners() {
        this.inputForm.addEventListener('submit', (e) => {
            e.preventDefault();
            this.sendMessage();
        });
        
        this.messageInput.addEventListener('input', () => {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = `${Math.min(this.messageInput.scrollHeight, 120)}px`;
            this.updateSendButtonState();
        });
        
        this.messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                const hasText = this.messageInput.value.trim().length > 0;
                if (hasText && !this.isProcessing) {
                    this.sendMessage();
                }
            }
        });
        
        // Initial button state
        this.updateSendButtonState();
        
        // New chat button
        this.newChatButton.addEventListener('click', () => {
            this.createNewContext();
        });
        
        // Agent selector
        this.agentSelect.addEventListener('change', (e) => {
            const agentId = e.target.value;
            if (agentId) {
                this.selectAgent(agentId);
            }
        });
        
        // Copy chat button
        if (this.copyChatButton) {
            this.copyChatButton.addEventListener('click', () => {
                this.copyEntireChat();
            });
        }
    }
    
    async loadAgents() {
        try {
            const response = await fetch('/api/agents');
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.agents = await response.json();
            this.renderAgentSelect();
            
            // Default to first agent or personal-assistant if available
            const defaultAgent = this.agents.find(a => a.id === 'personal-assistant') || this.agents[0];
            if (defaultAgent) {
                this.agentSelect.value = defaultAgent.id;
                await this.selectAgent(defaultAgent.id);
            }
        } catch (error) {
            console.error('Error loading agents:', error);
            this.agentSelect.innerHTML = '<option value="">Failed to load agents</option>';
        }
    }
    
    renderAgentSelect() {
        this.agentSelect.innerHTML = this.agents.map(agent => 
            `<option value="${agent.id}">${agent.name}</option>`
        ).join('');
    }
    
    async selectAgent(agentId) {
        const previousAgentId = this.currentAgentId;
        this.currentAgentId = agentId;
        
        // Clear current context if agent changed
        if (previousAgentId && previousAgentId !== agentId) {
            this.currentContextId = null;
            this.messagesContainer.innerHTML = '';
            this.messages = [];
        }
        
        try {
            // Load agent card
            const response = await fetch(`/api/agents/${agentId}/.well-known/agent-card.json`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.agentCard = await response.json();
            this.renderAgentCard();
            
            // Reload contexts for the new agent
            await this.loadAllContexts();
        } catch (error) {
            console.error('Error loading agent card:', error);
            this.agentCardContainer.innerHTML = '<div class="agent-card-error">Failed to load agent card</div>';
        }
    }
    
    renderAgentCard() {
        if (!this.agentCard) return;
        
        const skills = this.agentCard.skills || [];
        const skillsHtml = skills.map(skill => `
            <div class="skill-item">
                <div class="skill-name">${skill.name || skill.id}</div>
                <div class="skill-description">${skill.description || ''}</div>
                ${skill.tags && skill.tags.length > 0 ? `<div class="skill-tags">${skill.tags.map(tag => `<span class="skill-tag">${tag}</span>`).join('')}</div>` : ''}
            </div>
        `).join('');
        
        this.agentCardContainer.innerHTML = `
            <div class="agent-card">
                <div class="agent-card-header" id="agentCardHeader">
                    <div class="agent-card-header-content">
                        <h2 class="agent-card-name">${this.agentCard.name || 'Unknown Agent'}</h2>
                        <div class="agent-card-version">v${this.agentCard.version || '1.0.0'}</div>
                    </div>
                    <button class="agent-card-toggle" id="agentCardToggle" aria-label="Toggle agent card details">
                        <span class="toggle-icon">▼</span>
                    </button>
                </div>
                <div class="agent-card-body" id="agentCardBody">
                    <div class="agent-card-description">${this.agentCard.description || ''}</div>
                    ${skills.length > 0 ? `
                        <div class="agent-card-skills">
                            <h3>Capabilities</h3>
                            <div class="skills-list">
                                ${skillsHtml}
                            </div>
                        </div>
                    ` : ''}
                </div>
            </div>
        `;
        
        // Attach toggle event listener
        const toggleButton = document.getElementById('agentCardToggle');
        const cardHeader = document.getElementById('agentCardHeader');
        const cardBody = document.getElementById('agentCardBody');
        if (toggleButton && cardBody && cardHeader) {
            // Start collapsed by default
            cardBody.classList.add('collapsed');
            toggleButton.classList.add('collapsed');
            
            const toggleCard = () => {
                cardBody.classList.toggle('collapsed');
                toggleButton.classList.toggle('collapsed');
            };
            
            // Make entire header clickable
            cardHeader.addEventListener('click', (e) => {
                // Don't trigger if clicking directly on the toggle button (to avoid double-trigger)
                if (e.target !== toggleButton && !toggleButton.contains(e.target)) {
                    toggleCard();
                }
            });
            
            // Also allow clicking the button directly
            toggleButton.addEventListener('click', (e) => {
                e.stopPropagation();
                toggleCard();
            });
        }
    }
    
    async loadAllContexts() {
        // Contexts are agent-agnostic, load all contexts
        try {
            const response = await fetch(`/api/contexts`, {
                cache: 'no-cache', // Prevent caching
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            this.contexts = await response.json();
            console.log('Loaded contexts:', this.contexts.length, this.contexts.map(c => c.id));
            this.renderContextsList();
        } catch (error) {
            console.error('Error loading contexts:', error);
            this.contextsList.innerHTML = '<div class="context-item error">Failed to load contexts</div>';
        }
    }
    
    async loadContextChildren(contextId) {
        /** Load child contexts for a given context. */
        try {
            const response = await fetch(`/api/contexts?parent_id=${contextId}`, {
                cache: 'no-cache',
                headers: {
                    'Cache-Control': 'no-cache'
                }
            });
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error loading context children:', error);
            return [];
        }
    }
    
    renderContextsList() {
        if (this.contexts.length === 0) {
            this.contextsList.innerHTML = '<div class="context-item empty">No chats yet. Create a new one!</div>';
            // Clear current context if no contexts exist
            if (this.currentContextId) {
                this.currentContextId = null;
                this.messagesContainer.innerHTML = '';
                this.messages = [];
            }
            return;
        }
        
        // If current context is not in the list, clear it
        const currentContextExists = this.contexts.some(c => c.id === this.currentContextId);
        if (this.currentContextId && !currentContextExists) {
            this.currentContextId = null;
            this.messagesContainer.innerHTML = '';
            this.messages = [];
        }
        
        // Render root contexts (those without a parent)
        const rootContexts = this.contexts.filter(c => !c.parent_context_id);
        
        this.contextsList.innerHTML = rootContexts.map(context => {
            return this.renderContextWithChildren(context);
        }).join('');
        
        // Add click handlers
        this.contextsList.querySelectorAll('.context-item').forEach(item => {
            item.addEventListener('click', (e) => {
                // Don't trigger if clicking the delete button or expand button
                if (e.target.closest('.delete-context-btn') || e.target.closest('.expand-children-btn')) {
                    return;
                }
                const contextId = item.dataset.contextId;
                if (contextId) {
                    this.selectContext(contextId);
                }
            });
        });
        
        // Add expand/collapse handlers
        this.contextsList.querySelectorAll('.expand-children-btn').forEach(btn => {
            btn.addEventListener('click', async (e) => {
                e.stopPropagation();
                const contextId = btn.dataset.contextId;
                if (!contextId) return;
                
                const contextItem = btn.closest('.context-item');
                const childrenContainer = contextItem?.querySelector('.context-children');
                if (!contextItem || !childrenContainer) return;
                
                const isExpanded = childrenContainer.classList.contains('expanded');
                
                if (isExpanded) {
                    // Collapse
                    childrenContainer.classList.remove('expanded');
                    btn.textContent = '▶';
                    } else {
                        // Expand - load children if not already loaded
                        if (childrenContainer.dataset.loaded !== 'true') {
                            childrenContainer.innerHTML = '<div class="context-item loading">Loading...</div>';
                            const children = await this.loadContextChildren(contextId);
                            if (children.length > 0) {
                                childrenContainer.innerHTML = children.map(child => 
                                    this.renderContextWithChildren(child, true)
                                ).join('');
                                // Add click handlers for children
                                childrenContainer.querySelectorAll('.context-item').forEach(item => {
                                    item.addEventListener('click', (e) => {
                                        if (e.target.closest('.delete-context-btn') || e.target.closest('.expand-children-btn')) {
                                            return;
                                        }
                                        e.stopPropagation(); // Prevent event from bubbling to parent context
                                        const childContextId = item.dataset.contextId;
                                        console.log('[UI] Child context clicked:', childContextId, 'Item:', item);
                                        if (childContextId) {
                                            this.selectContext(childContextId);
                                        } else {
                                            console.error('[UI] No context ID found on child item:', item);
                                        }
                                    });
                                });
                                // Add expand handlers for children
                                childrenContainer.querySelectorAll('.expand-children-btn').forEach(childBtn => {
                                    childBtn.addEventListener('click', async (e) => {
                                        e.stopPropagation();
                                        const childContextId = childBtn.dataset.contextId;
                                        if (!childContextId) return;
                                        
                                        const childContextItem = childBtn.closest('.context-item');
                                        const childChildrenContainer = childContextItem?.querySelector('.context-children');
                                        if (!childContextItem || !childChildrenContainer) return;
                                        
                                        const isExpanded = childChildrenContainer.classList.contains('expanded');
                                        
                                        if (isExpanded) {
                                            childChildrenContainer.classList.remove('expanded');
                                            childBtn.textContent = '▶';
                                        } else {
                                            if (childChildrenContainer.dataset.loaded !== 'true') {
                                                childChildrenContainer.innerHTML = '<div class="context-item loading">Loading...</div>';
                                                const grandChildren = await this.loadContextChildren(childContextId);
                                                if (grandChildren.length > 0) {
                                                    childChildrenContainer.innerHTML = grandChildren.map(grandChild => 
                                                        this.renderContextWithChildren(grandChild, true)
                                                    ).join('');
                                                } else {
                                                    childChildrenContainer.innerHTML = '';
                                                }
                                                childChildrenContainer.dataset.loaded = 'true';
                                            }
                                            childChildrenContainer.classList.add('expanded');
                                            childBtn.textContent = '▼';
                                        }
                                    });
                                });
                            } else {
                                childrenContainer.innerHTML = '';
                                // Hide expand button if no children
                                btn.style.display = 'none';
                            }
                            childrenContainer.dataset.loaded = 'true';
                        }
                        childrenContainer.classList.add('expanded');
                        btn.textContent = '▼';
                    }
            });
        });
    }
    
    renderContextWithChildren(context, isChild = false) {
        /** Render a context with its children structure. */
        const isActive = context.id === this.currentContextId;
        const date = new Date(context.updated_at);
        const dateStr = date.toLocaleDateString();
        const childClass = isChild ? 'context-child' : '';
        
        return `
            <div class="context-item ${isActive ? 'active' : ''} ${childClass}" data-context-id="${context.id}">
                <div class="context-preview">
                    <button class="expand-children-btn" data-context-id="${context.id}" title="Expand children">▶</button>
                    <div class="context-info">
                        <div class="context-title">Chat ${dateStr}</div>
                        <div class="context-date">${date.toLocaleString()}</div>
                    </div>
                    <button class="delete-context-btn" data-context-id="${context.id}" title="Delete chat" onclick="event.stopPropagation(); app.deleteContext('${context.id}')">
                        <span>×</span>
                    </button>
                </div>
                <div class="context-children" data-context-id="${context.id}"></div>
            </div>
        `;
    }
    
    async createNewContext() {
        if (!this.currentAgentId) {
            alert('Please select an agent first');
            return;
        }
        try {
            const response = await fetch(`/api/agents/${this.currentAgentId}/contexts`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({})
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const context = await response.json();
            this.currentContextId = context.id;
            this.messagesContainer.innerHTML = '';
            this.messages = [];
            this.loadAllContexts();
            this.messageInput.focus();
        } catch (error) {
            console.error('Error creating context:', error);
            alert('Failed to create new chat. Please try again.');
        }
    }
    
    async deleteContext(contextId) {
        if (!confirm('Are you sure you want to delete this chat? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch(`/api/agents/${this.currentAgentId}/contexts/${contextId}`, {
                method: 'DELETE',
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(`HTTP error! status: ${response.status}, error: ${errorData.error || 'Unknown error'}`);
            }
            
            const result = await response.json();
            console.log('Context deleted successfully', result);
            
            // If we deleted the current context, clear it first
            const wasCurrentContext = this.currentContextId === contextId;
            if (wasCurrentContext) {
                this.currentContextId = null;
                this.messagesContainer.innerHTML = '';
                this.messages = [];
            }
            
            // Remove from local array immediately
            this.contexts = this.contexts.filter(c => c.id !== contextId);
            
            // Re-render the list immediately
            this.renderContextsList();
            
            // Reload from server to ensure consistency (but don't create a new context)
            await this.loadAllContexts();
            
            // If we deleted the current context and there are no contexts left, 
            // the user can create a new one manually - don't auto-create
        } catch (error) {
            console.error('Error deleting context:', error);
            alert(`Failed to delete chat: ${error.message}. Please try again.`);
        }
    }
    
    async selectContext(contextId) {
        console.log('[UI] Selecting context:', contextId);
        this.currentContextId = contextId;
        this.renderContextsList();
        
        try {
            // Use agent-agnostic endpoint - contexts are not tied to specific agents
            const url = `/api/contexts/${contextId}/messages`;
            console.log('[UI] Fetching messages from:', url);
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('[UI] Received messages for context:', contextId, 'Message count:', data.messages?.length || 0);
            this.messagesContainer.innerHTML = '';
            this.messages = [];
            
            // First pass: build a map of tool_call_id -> whether it has a corresponding tool result
            const toolCallIdsWithResults = new Set();
            data.messages.forEach((msg, index) => {
                if (msg.role === 'tool' && msg.toolName) {
                    // Find the corresponding assistant message with this tool call
                    // Look backwards to find the assistant message with matching tool_call_id
                    for (let i = index - 1; i >= 0; i--) {
                        const prevMsg = data.messages[i];
                        if (prevMsg.role === 'agent' && prevMsg.tool_calls && Array.isArray(prevMsg.tool_calls)) {
                            // Check if any tool call matches this tool result
                            const matchingToolCall = prevMsg.tool_calls.find(tc => {
                                // We need to match by tool name since we don't have tool_call_id in the response
                                // Check if the tool name matches and it's the right position
                                return tc.function?.name === msg.toolName;
                            });
                            if (matchingToolCall && matchingToolCall.id) {
                                toolCallIdsWithResults.add(matchingToolCall.id);
                                break;
                            }
                        }
                    }
                }
            });
            
            // Second pass: render messages
            data.messages.forEach((msg, index) => {
                console.log(`[UI] Rendering message ${index}:`, { 
                    role: msg.role, 
                    hasMessageId: !!msg.messageId, 
                    messageId: msg.messageId, 
                    toolName: msg.toolName,
                    hasToolName: !!msg.toolName,
                    content: msg.content?.substring(0, 50),
                    fullMsg: msg
                });
                
                // Check if this is a tool message (role === 'tool')
                if (msg.role === 'tool') {
                    if (msg.toolName) {
                        console.log(`[UI] Adding tool result message ${index}:`, msg.toolName, msg.content?.substring(0, 50));
                        this.addToolResultMessage(msg.toolName, msg.content || '', msg.messageId);
                    } else {
                        console.warn(`[UI] Tool message ${index} missing toolName, adding anyway with role as toolName`);
                        // Fallback: use a generic tool name if missing
                        this.addToolResultMessage('tool', msg.content || '', msg.messageId);
                    }
                }
                // Check if this is an assistant message with tool calls - render as tool approval
                else if (msg.role === 'agent' && msg.tool_calls && Array.isArray(msg.tool_calls) && msg.tool_calls.length > 0) {
                    // Render tool approval for each tool call
                    // Check if there's a corresponding tool result message after this
                    msg.tool_calls.forEach(toolCall => {
                        const toolName = toolCall.function?.name || 'unknown';
                        const toolArgs = toolCall.function?.arguments || '{}';
                        const isApproved = toolCall.id && toolCallIdsWithResults.has(toolCall.id);
                        const approvalText = isApproved 
                            ? `Tool call approved:\nTool: ${toolName}\nArguments: ${toolArgs}`
                            : `Tool call requires approval:\nTool: ${toolName}\nArguments: ${toolArgs}`;
                        this.addToolApprovalMessage(approvalText, msg.taskId || '');
                    });
                }
                // Check if this is a tool approval message (legacy format)
                else if (msg.content && (msg.content.includes('Tool call requires approval:') || msg.content.includes('Tool call approved:'))) {
                    this.addToolApprovalMessage(msg.content, msg.taskId || '');
                } 
                // Regular message - only add if it has content
                else if (msg.content && msg.content.trim()) {
                    this.addMessage(msg.role, msg.content, false, msg.messageId);
                }
            });
            
            console.log(`[UI] Finished rendering ${data.messages.length} messages. Container now has ${this.messagesContainer.children.length} children.`);
            
            this.scrollToBottom();
        } catch (error) {
            console.error('Error loading context messages:', error);
            this.messagesContainer.innerHTML = '<div class="message agent"><div class="message-content"><p>Failed to load messages. Please try again.</p></div></div>';
        }
    }
    
    updateStatus(state, text) {
        this.statusIndicator.className = `status-indicator ${state}`;
        this.statusText.textContent = text;
    }
    
    updateSendButtonState() {
        const hasText = this.messageInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText || this.isProcessing;
    }
    
    async sendMessage() {
        const text = this.messageInput.value.trim();
        if (!text || this.isProcessing) return;
        
        if (!this.currentAgentId) {
            alert('Please select an agent first');
            return;
        }
        
        // Create context if none exists
        if (!this.currentContextId) {
            await this.createNewContext();
            if (!this.currentContextId) {
                // Failed to create context
                return;
            }
        }
        
        this.isProcessing = true;
        this.messageInput.value = '';
        this.messageInput.style.height = 'auto';
        this.updateSendButtonState();
        
        // Generate messageId for user message
        const userMessageId = `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        this.addMessage('user', text, false, userMessageId);
        this.updateStatus('connecting', 'Sending...');
        
        try {
            // A2A SDK uses JSON-RPC 2.0 format
            const requestId = Date.now().toString();
            const response = await fetch(`/api/agents/${this.currentAgentId}/`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    jsonrpc: '2.0',
                    method: 'message/stream',
                    params: {
                        message: {
                            messageId: requestId,
                            role: 'user',
                            parts: [{ kind: 'text', text: text }],
                            contextId: this.currentContextId
                        }
                    },
                    id: requestId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            // Check if response is SSE stream
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('text/event-stream')) {
                // Handle SSE stream
                this.updateStatus('connecting', 'Processing...');
                this.handleSSEStream(response);
            } else {
                // Handle JSON response
                const result = await response.json();
                if (result.error) {
                    throw new Error(result.error.message || 'Unknown error');
                }
                
                // If we get a task ID, connect to stream
                if (result.result && result.result.id) {
                    this.currentTaskId = result.result.id;
                    this.updateStatus('connecting', 'Processing...');
                    this.connectToStream();
                } else {
                    // Direct response
                    if (result.result && result.result.message) {
                        const content = this.extractTextFromMessage(result.result.message);
                        if (content) {
                            this.addMessage('agent', content);
                        }
                    }
                    this.updateStatus('ready', 'Ready');
                    this.isProcessing = false;
                    this.updateSendButtonState();
                }
            }
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('agent', `Error: ${error.message}`);
            this.updateStatus('error', 'Error');
            this.isProcessing = false;
            this.updateSendButtonState();
        }
    }
    
    async handleSSEStream(response) {
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        try {
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';
                
                for (const line of lines) {
                    if (line.startsWith('data: ')) {
                        try {
                            const data = JSON.parse(line.slice(6));
                            console.log('Received SSE event:', data);
                            this.handleStreamEvent(data);
                        } catch (error) {
                            console.error('Error parsing SSE data:', error);
                        }
                    }
                }
            }
        } catch (error) {
            console.error('Error reading SSE stream:', error);
            this.updateStatus('error', 'Stream error');
            this.isProcessing = false;
            this.updateSendButtonState();
        } finally {
            reader.releaseLock();
        }
    }
    
    connectToStream() {
        if (!this.currentTaskId) return;
        
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        // A2A SDK streams via the same POST endpoint, but we'll use a GET endpoint if available
        // For now, streaming happens in the initial POST response
        // This method is kept for compatibility but may not be needed
        this.updateStatus('ready', 'Ready');
        this.isProcessing = false;
        this.updateSendButtonState();
    }
    
    handleStreamEvent(data) {
        // Handle JSON-RPC response format
        if (data.jsonrpc === '2.0' && data.result) {
            const result = data.result;
            
            if (result.kind === 'task') {
                this.currentTaskId = result.id;
                const newContextId = result.contextId || this.currentContextId;
                if (newContextId !== this.currentContextId) {
                    this.currentContextId = newContextId;
                    // Refresh contexts list to show new context
                    this.loadAllContexts();
                }
                this.updateStatus('connecting', 'Task created');
            } else if (result.kind === 'status-update') {
                const status = result.status;
                const state = status.state;
                console.log('Status update received', { state, hasMessage: !!status.message, final: result.final });
                
                if (state === 'working') {
                    this.updateStatus('connecting', 'Processing...');
                    if (status.message) {
                        const content = this.extractTextFromMessage(status.message);
                        const messageRole = status.message.role;
                        console.log('Working state message:', { content, role: messageRole, final: result.final });
                        
                        if (content) {
                            // Check if this is a tool message (role === 'tool')
                            if (messageRole === 'tool') {
                                const toolName = status.message.toolName;
                                console.log('Tool message received', { toolName, content: content.substring(0, 100), fullMessage: JSON.stringify(status.message) });
                                if (toolName) {
                                    // Remove processing message before showing tool result
                                    this.removeProcessingMessage();
                                    this.addToolResultMessage(toolName, content);
                                } else {
                                    console.warn('Tool message missing toolName', { message: status.message, keys: Object.keys(status.message || {}) });
                                }
                            } else if (content === 'Processing your request...') {
                                // Only add if we don't already have a more specific message
                                // Check if we already have a tool approval or tool result
                                const hasToolApproval = this.messagesContainer.querySelector('.tool-approval');
                                const hasToolResult = this.messagesContainer.querySelector('.tool-result');
                                if (!hasToolApproval && !hasToolResult) {
                                    // Generate messageId for processing message if not present
                                    const messageId = status.message?.messageId || `processing-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                                    this.updateOrAddAgentMessage(content, !result.final, messageId);
                                }
                            } else {
                                // Regular working message - remove processing message first
                                this.removeProcessingMessage();
                                // Regular working message - can be updated by streaming
                                // Use messageId to track specific messages (important for child messages)
                                // Generate messageId if not present
                                const messageId = status.message?.messageId || `agent-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
                                this.updateOrAddAgentMessage(content, !result.final, messageId);
                            }
                        }
                    }
                } else if (state === 'completed') {
                    this.removeProcessingMessage();
                    console.log('Received completed state', { status, result });
                    this.updateStatus('ready', 'Ready');
                    if (status.message) {
                        const content = this.extractTextFromMessage(status.message);
                        console.log('Completed message content:', content);
                        if (content) {
                            // Use messageId to track completed messages separately
                            const messageId = status.message.messageId;
                            this.updateOrAddAgentMessage(content, false, messageId);
                        }
                    } else {
                        console.warn('Completed state but no message in status');
                    }
                    this.isProcessing = false;
                    this.updateSendButtonState();
                    this.currentTaskId = null;
                } else if (state === 'failed') {
                    this.removeProcessingMessage();
                    this.updateStatus('error', 'Error');
                    const errorMsg = status.message 
                        ? this.extractTextFromMessage(status.message)
                        : 'Task failed';
                    this.addMessage('agent', errorMsg);
                    this.isProcessing = false;
                    this.updateSendButtonState();
                    this.currentTaskId = null;
                } else if (state === 'input-required') {
                    this.removeProcessingMessage();
                    this.updateStatus('connecting', 'Approval required');
                    if (status.message) {
                        const content = this.extractTextFromMessage(status.message);
                        if (content) {
                            // Check if this is a tool approval request
                            if (content.includes('Tool call requires approval:')) {
                                this.addToolApprovalMessage(content, result.taskId);
                            } else {
                                this.updateOrAddAgentMessage(content, false);
                            }
                        }
                    }
                }
            }
        } else {
            // Legacy format support (if needed)
            if (data.kind === 'task') {
                this.currentTaskId = data.id;
                this.updateStatus('connecting', 'Task created');
            } else if (data.kind === 'status-update') {
                const status = data.status;
                const state = status.state;
                
                if (state === 'working') {
                    this.updateStatus('connecting', 'Processing...');
                    if (status.message) {
                        const content = this.extractTextFromMessage(status.message);
                        if (content) {
                            this.updateOrAddAgentMessage(content, true);
                        }
                    }
                } else if (state === 'completed') {
                    this.updateStatus('ready', 'Ready');
                    if (status.message) {
                        const content = this.extractTextFromMessage(status.message);
                        if (content) {
                            this.updateOrAddAgentMessage(content, false);
                        }
                    }
                    this.isProcessing = false;
                    this.updateSendButtonState();
                    this.currentTaskId = null;
                } else if (state === 'failed') {
                    this.updateStatus('error', 'Error');
                    const errorMsg = status.message 
                        ? this.extractTextFromMessage(status.message)
                        : 'Task failed';
                    this.addMessage('agent', errorMsg);
                    this.isProcessing = false;
                    this.updateSendButtonState();
                    this.currentTaskId = null;
                }
            }
        }
    }
    
    extractTextFromMessage(message) {
        if (typeof message === 'string') {
            return message;
        }
        
        if (message.parts && Array.isArray(message.parts)) {
            return message.parts
                .filter(part => part.kind === 'text' && part.text)
                .map(part => part.text)
                .join('');
        }
        
        if (message.content) {
            return message.content;
        }
        
        return null;
    }
    
    removeProcessingMessage() {
        /** Remove any "Processing your request..." messages. */
        const messages = this.messagesContainer.querySelectorAll('.message.agent');
        messages.forEach(msg => {
            const contentEl = msg.querySelector('.message-content p');
            if (contentEl && contentEl.textContent === 'Processing your request...') {
                msg.remove();
            }
        });
    }
    
    updateOrAddAgentMessage(content, isStreaming, messageId = null) {
        // If starting to stream a new message, remove streaming class from all other messages
        if (isStreaming && messageId) {
            // Remove streaming class from all messages except the one we're about to update/create
            Array.from(this.messagesContainer.children).forEach(msg => {
                if (msg.dataset.messageId !== messageId) {
                    msg.classList.remove('streaming');
                }
            });
        }
        
        // ALWAYS use messageId if provided - never update a message without messageId when we have one
        if (messageId) {
            const existingMessage = Array.from(this.messagesContainer.children).find(msg => {
                return msg.dataset.messageId === messageId;
            });
            if (existingMessage) {
                const contentEl = existingMessage.querySelector('.message-content p');
                if (contentEl) {
                    contentEl.textContent = content;
                    // Explicitly add or remove streaming class
                    if (isStreaming) {
                        existingMessage.classList.add('streaming');
                    } else {
                        existingMessage.classList.remove('streaming');
                    }
                    // Ensure copy icon exists if messageId is present
                    const contentDiv = existingMessage.querySelector('.message-content');
                    if (contentDiv && !contentDiv.querySelector('.message-copy-icon')) {
                        const copyIcon = document.createElement('button');
                        copyIcon.className = 'message-copy-icon';
                        copyIcon.innerHTML = '📋';
                        copyIcon.title = 'Copy OpenAI message';
                        copyIcon.onclick = async (e) => {
                            e.stopPropagation();
                            await this.copyOpenAIMessage(messageId, e);
                        };
                        contentDiv.appendChild(copyIcon);
                    }
                    this.scrollToBottom();
                    return;
                }
            }
            // MessageId provided but message not found - create new message with that ID
            this.addMessage('agent', content, isStreaming, messageId);
            return;
        }
        
        // No messageId provided - check if we should update last message or create new one
        const lastMessage = this.messagesContainer.lastElementChild;
        if (lastMessage && lastMessage.classList.contains('agent')) {
            const lastMessageId = lastMessage.dataset.messageId;
            const wasStreaming = lastMessage.classList.contains('streaming');
            const contentEl = lastMessage.querySelector('.message-content p');
            
            // If last message has a messageId, it's a completed message from another agent - create new message
            if (lastMessageId) {
                this.addMessage('agent', content, isStreaming, messageId);
                return;
            }
            
            // Only update if last message has NO messageId (legacy/processing message) AND is streaming
            if (contentEl && !lastMessageId && wasStreaming && isStreaming) {
                // If updating a "Processing your request..." message, replace it
                if (contentEl.textContent === 'Processing your request...' && content !== 'Processing your request...') {
                    contentEl.textContent = content;
                    // Explicitly add or remove streaming class
                    if (isStreaming) {
                        lastMessage.classList.add('streaming');
                    } else {
                        lastMessage.classList.remove('streaming');
                    }
                    this.scrollToBottom();
                    return;
                }
                // Only update if it's currently streaming (same message being updated)
                contentEl.textContent = content;
                // Explicitly add or remove streaming class
                if (isStreaming) {
                    lastMessage.classList.add('streaming');
                } else {
                    lastMessage.classList.remove('streaming');
                }
                this.scrollToBottom();
                return;
            }
        }
        
        // Otherwise, add a new message
        this.addMessage('agent', content, isStreaming, messageId);
    }
    
    addMessage(role, content, isStreaming = false, messageId = null) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        if (isStreaming) {
            messageDiv.classList.add('streaming');
        }
        if (messageId) {
            messageDiv.dataset.messageId = messageId;
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        const p = document.createElement('p');
        p.textContent = content;
        contentDiv.appendChild(p);
        
        // Add copy icon if messageId exists
        if (messageId) {
            const copyIcon = document.createElement('button');
            copyIcon.className = 'message-copy-icon';
            copyIcon.innerHTML = '📋';
            copyIcon.title = 'Copy OpenAI message';
            copyIcon.onclick = async (e) => {
                e.stopPropagation();
                await this.copyOpenAIMessage(messageId, e);
            };
            contentDiv.appendChild(copyIcon);
            console.log('Added copy icon to message', { messageId, role });
        } else {
            console.log('No messageId for message, skipping copy icon', { role, content: content.substring(0, 50) });
        }
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        this.scrollToBottom();
    }
    
    async copyOpenAIMessage(messageId, event) {
        try {
            const contextId = this.currentContextId;
            if (!contextId) {
                console.error('No context ID available');
                return;
            }
            
            // Check if this is a client-generated messageId (starts with 'user-', 'processing-', 'agent-', or 'tool-')
            // These won't exist in the database yet
            if (messageId.startsWith('user-') || messageId.startsWith('processing-') || 
                messageId.startsWith('agent-') || messageId.startsWith('tool-')) {
                // For client-generated IDs, try to get the message from the DOM
                const messageEl = Array.from(this.messagesContainer.children).find(
                    el => el.dataset.messageId === messageId
                );
                
                if (messageEl) {
                    const contentEl = messageEl.querySelector('.message-content p');
                    const content = contentEl?.textContent || '';
                    const role = messageEl.classList.contains('user') ? 'user' : 
                               messageEl.classList.contains('tool-result') ? 'tool' : 'assistant';
                    
                    // Create a simple message object
                    const messageObj = {
                        role: role,
                        content: content
                    };
                    
                    // If it's a tool result, try to get the tool name
                    if (messageEl.classList.contains('tool-result')) {
                        const toolResultTitle = messageEl.querySelector('.tool-result-title');
                        const toolName = toolResultTitle?.textContent?.replace('🔧 Tool Result: ', '') || '';
                        messageObj.name = toolName;
                    }
                    
                    await navigator.clipboard.writeText(JSON.stringify(messageObj, null, 2));
                    console.log('Copied client-generated message to clipboard:', messageObj);
                } else {
                    console.warn('Message not found in DOM:', messageId);
                    return;
                }
            } else {
                // Server-generated messageId - fetch from API
                const response = await fetch(`/api/contexts/${contextId}/messages/${messageId}`);
                if (!response.ok) {
                    if (response.status === 404) {
                        console.warn('Message not found in database:', messageId);
                        // Try to get from DOM as fallback
                        const messageEl = Array.from(this.messagesContainer.children).find(
                            el => el.dataset.messageId === messageId
                        );
                        if (messageEl) {
                            const contentEl = messageEl.querySelector('.message-content p');
                            const content = contentEl?.textContent || '';
                            const role = messageEl.classList.contains('user') ? 'user' : 
                                       messageEl.classList.contains('tool-result') ? 'tool' : 'assistant';
                            const messageObj = { role, content };
                            await navigator.clipboard.writeText(JSON.stringify(messageObj, null, 2));
                            console.log('Copied message from DOM as fallback:', messageObj);
                        } else {
                            throw new Error('Message not found');
                        }
                    } else {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return;
                }
                
                const data = await response.json();
                const messageToCopy = data.message;
                
                if (messageToCopy) {
                    await navigator.clipboard.writeText(JSON.stringify(messageToCopy, null, 2));
                    console.log('Copied OpenAI message to clipboard:', messageToCopy);
                } else {
                    console.warn('No message found to copy');
                    return;
                }
            }
            
            // Show visual feedback
            const copyButton = event?.target;
            if (copyButton) {
                const originalText = copyButton.innerHTML;
                copyButton.innerHTML = '✓';
                copyButton.style.color = '#4caf50';
                setTimeout(() => {
                    copyButton.innerHTML = originalText;
                    copyButton.style.color = '';
                }, 1000);
            }
        } catch (error) {
            console.error('Error copying OpenAI message:', error);
            // Don't show alert for 404s - just log
            if (!error.message?.includes('not found')) {
                alert(`Failed to copy message: ${error.message}`);
            }
        }
    }
    
    async copyEntireChat() {
        try {
            const contextId = this.currentContextId;
            if (!contextId) {
                alert('No chat selected');
                return;
            }
            
            // Fetch messages from the server - same request as selectContext
            const response = await fetch(`/api/contexts/${contextId}/messages`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            const messages = data.messages || [];
            
            if (messages.length === 0) {
                alert('No messages to copy');
                return;
            }
            
            // Copy the messages array as-is from the server response
            const chatJson = JSON.stringify(messages, null, 2);
            
            await navigator.clipboard.writeText(chatJson);
            
            // Show visual feedback
            const originalText = this.copyChatButton.innerHTML;
            this.copyChatButton.innerHTML = '✓ Copied!';
            this.copyChatButton.style.color = '#4caf50';
            setTimeout(() => {
                this.copyChatButton.innerHTML = originalText;
                this.copyChatButton.style.color = '';
            }, 2000);
        } catch (error) {
            console.error('Error copying chat:', error);
            alert(`Failed to copy chat: ${error.message}`);
        }
    }
    
    addToolResultMessage(toolName, result, messageId = null) {
        /** Add a distinct message showing tool execution result. */
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message agent tool-result';
        if (messageId) {
            messageDiv.dataset.messageId = messageId;
        }
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content tool-result-content';
        
        const title = document.createElement('div');
        title.className = 'tool-result-title';
        title.textContent = `🔧 Tool Result: ${toolName}`;
        contentDiv.appendChild(title);
        
        const resultDiv = document.createElement('div');
        resultDiv.className = 'tool-result-text';
        
        // Try to parse as JSON for pretty display, otherwise show as-is
        let displayResult = result;
        try {
            const parsed = JSON.parse(result);
            displayResult = JSON.stringify(parsed, null, 2);
        } catch {
            // Not JSON, use as-is
        }
        
        const pre = document.createElement('pre');
        pre.textContent = displayResult;
        resultDiv.appendChild(pre);
        contentDiv.appendChild(resultDiv);
        
        // Add copy icon if messageId exists
        if (messageId) {
            const copyIcon = document.createElement('button');
            copyIcon.className = 'message-copy-icon';
            copyIcon.innerHTML = '📋';
            copyIcon.title = 'Copy OpenAI message';
            copyIcon.onclick = async (e) => {
                e.stopPropagation();
                await this.copyOpenAIMessage(messageId, e);
            };
            contentDiv.appendChild(copyIcon);
        }
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        this.scrollToBottom();
    }
    
    addToolApprovalMessage(content, taskId) {
        // Parse the tool approval message - check if it's already approved
        const isApproved = content.includes('Tool call approved:');
        const toolMatch = content.match(/Tool: (.+)/);
        const argsMatch = content.match(/Arguments: (.+)/s);
        
        const toolName = toolMatch ? toolMatch[1].trim() : 'unknown';
        let toolArgs = {};
        if (argsMatch) {
            try {
                toolArgs = JSON.parse(argsMatch[1].trim());
            } catch (e) {
                // If parsing fails, show raw text
                toolArgs = { raw: argsMatch[1].trim() };
            }
        }
        
        // Check if we already have a pending approval for this tool call (only if not approved)
        if (!isApproved) {
            const existingApprovals = this.messagesContainer.querySelectorAll('.tool-approval');
            for (const approval of existingApprovals) {
                const approvalTaskId = approval.dataset.taskId;
                const approvalButtons = approval.querySelector('.tool-approval-buttons');
                const hasButtons = approvalButtons && approvalButtons.querySelector('button');
                
                // If this approval is for the same task and still has buttons (not yet approved/rejected), skip creating a duplicate
                if (approvalTaskId === taskId && hasButtons) {
                    // Check if it's the same tool
                    const existingToolName = approval.querySelector('.tool-name code')?.textContent;
                    if (existingToolName === toolName) {
                        // Duplicate approval request, don't create another
                        return;
                    }
                }
            }
        }
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'message agent tool-approval';
        messageDiv.dataset.taskId = taskId;
        messageDiv.dataset.contextId = this.currentContextId || '';
        messageDiv.dataset.toolName = toolName;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content tool-approval-content';
        
        const title = document.createElement('div');
        title.className = 'tool-approval-title';
        title.textContent = isApproved ? '✓ Tool Call Approved' : '🔧 Tool Call Requires Approval';
        contentDiv.appendChild(title);
        
        const toolInfo = document.createElement('div');
        toolInfo.className = 'tool-info';
        
        const toolNameEl = document.createElement('div');
        toolNameEl.className = 'tool-name';
        toolNameEl.innerHTML = `<strong>Tool:</strong> <code>${toolName}</code>`;
        toolInfo.appendChild(toolNameEl);
        
        const toolArgsEl = document.createElement('div');
        toolArgsEl.className = 'tool-args';
        toolArgsEl.innerHTML = `<strong>Arguments:</strong> <pre>${JSON.stringify(toolArgs, null, 2)}</pre>`;
        toolInfo.appendChild(toolArgsEl);
        
        contentDiv.appendChild(toolInfo);
        
        // Only show buttons if not already approved
        if (!isApproved) {
            const buttonContainer = document.createElement('div');
            buttonContainer.className = 'tool-approval-buttons';
            
            const approveBtn = document.createElement('button');
            approveBtn.className = 'approve-btn';
            approveBtn.textContent = '✓ Approve';
            approveBtn.onclick = () => this.handleToolApproval(taskId, true);
            
            const rejectBtn = document.createElement('button');
            rejectBtn.className = 'reject-btn';
            rejectBtn.textContent = '✗ Reject';
            rejectBtn.onclick = () => this.handleToolApproval(taskId, false);
            
            buttonContainer.appendChild(approveBtn);
            buttonContainer.appendChild(rejectBtn);
            contentDiv.appendChild(buttonContainer);
        } else {
            // Show approved status
            const statusDiv = document.createElement('div');
            statusDiv.className = 'approval-status approved';
            statusDiv.textContent = '✓ Approved';
            contentDiv.appendChild(statusDiv);
        }
        
        messageDiv.appendChild(contentDiv);
        this.messagesContainer.appendChild(messageDiv);
        
        this.scrollToBottom();
    }
    
    async handleToolApproval(taskId, approved) {
        const approvalMessages = this.messagesContainer.querySelectorAll('.tool-approval');
        const lastApproval = approvalMessages[approvalMessages.length - 1];
        if (lastApproval) {
            const buttons = lastApproval.querySelector('.tool-approval-buttons');
            if (buttons) {
                buttons.innerHTML = `<div class="approval-status ${approved ? 'approved' : 'rejected'}">${approved ? '✓ Approved' : '✗ Rejected'}</div>`;
            }
        }
        
        // Get contextId and taskId from the approval message element (most reliable)
        const approvalContextId = lastApproval?.dataset.contextId || this.currentContextId || '';
        const approvalTaskId = lastApproval?.dataset.taskId || taskId || this.currentTaskId;
        
        console.log('[UI] Sending tool approval', {
            approved,
            approvalContextId,
            approvalTaskId,
            taskIdParam: taskId,
            currentTaskId: this.currentTaskId,
            currentContextId: this.currentContextId
        });
        
        // If currentTaskId is null and approvalTaskId is also null/empty, we need to find the active task
        // This can happen when loading messages from the database after a page refresh
        if (!this.currentTaskId && !approvalTaskId) {
            // Try to find an active task for this context by checking if there's a pending approval
            // If we can't find one, the task may have completed or failed
            console.warn('Cannot send approval: no taskId available. Task may have completed or failed.');
            this.addMessage('agent', 'The previous task has ended. Please try your request again.');
            return;
        }
        
        // If we have an approvalTaskId but no currentTaskId, set currentTaskId so we can track it
        if (approvalTaskId && !this.currentTaskId) {
            this.currentTaskId = approvalTaskId;
        }
        
        // Send approval via the same JSON-RPC endpoint
        try {
            const requestId = Date.now().toString();
            const finalTaskId = approvalTaskId || this.currentTaskId;
            const finalContextId = approvalContextId || this.currentContextId || '';
            
            console.log('[UI] Approval request payload', {
                contextId: finalContextId,
                taskId: finalTaskId,
                approved: approved ? 'approve' : 'reject'
            });
            
            let response;
            try {
                response = await fetch(`/api/agents/${this.currentAgentId}/`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        jsonrpc: '2.0',
                        method: 'message/stream',
                        params: {
                            message: {
                                messageId: requestId,
                                role: 'user',
                                parts: [{ kind: 'text', text: approved ? 'approve' : 'reject' }],
                                contextId: finalContextId,
                                taskId: finalTaskId, // Use the taskId from the approval element
                            }
                        },
                        id: requestId
                    })
                });
            } catch (fetchError) {
                // Handle network errors (e.g., task has completed, connection closed)
                if (fetchError instanceof TypeError && fetchError.message.includes('NetworkError')) {
                    console.warn('Network error sending approval (task may have completed):', fetchError);
                    this.addMessage('agent', 'The task has already completed. The approval cannot be processed.');
                    // Revert the approval button state
                    if (lastApproval) {
                        const buttons = lastApproval.querySelector('.tool-approval-buttons');
                        if (buttons) {
                            buttons.innerHTML = `
                                <button class="approve-btn" onclick="app.handleToolApproval('${taskId}', true)">✓ Approve</button>
                                <button class="reject-btn" onclick="app.handleToolApproval('${taskId}', false)">✗ Reject</button>
                            `;
                        }
                    }
                    return;
                }
                throw fetchError; // Re-throw if it's not a network error
            }
            
            if (!response.ok) {
                const errorText = await response.text();
                let errorMessage = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = JSON.parse(errorText);
                    if (errorData.error?.message) {
                        errorMessage = errorData.error.message;
                    }
                } catch {
                    // Use default error message
                }
                
                // If the task is in a terminal state, clear it and inform the user
                if (errorMessage.includes('terminal state') || errorMessage.includes('cannot be modified')) {
                    console.warn('Task is in terminal state, clearing current task');
                    this.currentTaskId = null;
                    this.addMessage('agent', 'The previous task has ended. Please try your request again.');
                    // Revert the approval button state
                    if (lastApproval) {
                        const buttons = lastApproval.querySelector('.tool-approval-buttons');
                        if (buttons) {
                            buttons.innerHTML = `
                                <button class="approve-btn" onclick="app.handleToolApproval('${taskId}', true)">✓ Approve</button>
                                <button class="reject-btn" onclick="app.handleToolApproval('${taskId}', false)">✗ Reject</button>
                            `;
                        }
                    }
                    return;
                }
                
                throw new Error(errorMessage);
            }
            
            // Handle the response stream
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('text/event-stream')) {
                this.updateStatus('connecting', 'Processing approval...');
                // Set currentTaskId so the stream handler knows which task to track
                if (finalTaskId && !this.currentTaskId) {
                    this.currentTaskId = finalTaskId;
                }
                // Handle SSE stream, but don't fail if it errors (task might have already completed)
                this.handleSSEStream(response).catch(error => {
                    console.warn('Error reading SSE stream for approval (task may have completed):', error);
                    // Don't show error to user - the approval may have been processed
                    this.updateStatus('ready', 'Ready');
                });
            } else {
                const result = await response.json();
                if (result.error) {
                    // Check for terminal state error in JSON-RPC response
                    if (result.error.message?.includes('terminal state') || result.error.message?.includes('cannot be modified')) {
                        console.warn('Task is in terminal state, clearing current task');
                        this.currentTaskId = null;
                        this.addMessage('agent', 'The previous task has ended. Please try your request again.');
                        // Revert the approval button state
                        if (lastApproval) {
                            const buttons = lastApproval.querySelector('.tool-approval-buttons');
                            if (buttons) {
                                buttons.innerHTML = `
                                    <button class="approve-btn" onclick="app.handleToolApproval('${taskId}', true)">✓ Approve</button>
                                    <button class="reject-btn" onclick="app.handleToolApproval('${taskId}', false)">✗ Reject</button>
                                `;
                            }
                        }
                        return;
                    }
                    throw new Error(result.error.message || 'Unknown error');
                }
            }
        } catch (error) {
            console.error('Error sending approval:', error);
            // Don't show alert for terminal state errors (already handled above)
            if (!error.message?.includes('terminal state') && !error.message?.includes('cannot be modified')) {
                alert(`Failed to send approval: ${error.message || 'Unknown error'}. Please try again.`);
            }
        }
    }
    
    scrollToBottom() {
        this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
    }
}

// Initialize the app when DOM is ready
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new ChatApp();
    window.app = app;
});

