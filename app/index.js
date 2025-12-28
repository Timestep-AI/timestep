const BASE_URL_KEY = 'agentCardBaseUrl';

// Load saved base URL
const savedBaseUrl = localStorage.getItem(BASE_URL_KEY);
if (savedBaseUrl) {
    document.getElementById('baseUrl').value = savedBaseUrl;
}

// Fetch agent card
async function fetchAgentCard() {
    const baseUrl = document.getElementById('baseUrl').value.trim();
    
    if (!baseUrl) {
        showError('Please enter a base URL');
        return;
    }
    
    // Save base URL
    localStorage.setItem(BASE_URL_KEY, baseUrl);
    
    // Normalize URL (remove trailing slash)
    const normalizedUrl = baseUrl.replace(/\/$/, '');
    const cardUrl = `${normalizedUrl}/.well-known/agent-card.json`;
    
    // Show loading
    document.getElementById('loading').classList.remove('hidden');
    document.getElementById('error').classList.add('hidden');
    document.getElementById('cardContainer').classList.add('hidden');
    
    try {
        const response = await fetch(cardUrl, {
            method: 'GET',
            headers: {
                'Accept': 'application/json',
            },
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const card = await response.json();
        
        // Display card
        document.getElementById('cardJson').textContent = JSON.stringify(card, null, 2);
        document.getElementById('cardContainer').classList.remove('hidden');
        document.getElementById('loading').classList.add('hidden');
        
    } catch (error) {
        showError(`Failed to fetch agent card: ${error.message}`);
        document.getElementById('loading').classList.add('hidden');
    }
}

function showError(message) {
    const errorDiv = document.getElementById('error');
    errorDiv.textContent = message;
    errorDiv.classList.remove('hidden');
}

// Event listeners
document.getElementById('fetchBtn').addEventListener('click', fetchAgentCard);

document.getElementById('baseUrl').addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        fetchAgentCard();
    }
});

// Fetch on page load if URL is set
if (savedBaseUrl) {
    fetchAgentCard();
}

