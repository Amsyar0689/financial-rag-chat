const chatHistory = document.getElementById('chatHistory');
const userInput = document.getElementById('userInput');
const modal = document.getElementById('sourceModal');
const pdfViewer = document.getElementById('pdfViewer');
const modalHeader = document.getElementById('modalHeader');

const PDF_BASE_URL = "https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/c24e7a28-5254-4dfa-9447-62aaa3c24bb1.pdf";

function handleEnter(e) {
    if (e.key === 'Enter') sendMessage();
}

async function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    addMessage(text, 'user-msg');
    userInput.value = '';

    const aiMessageDiv = document.createElement('div');
    aiMessageDiv.className = 'message ai-msg';
    chatHistory.appendChild(aiMessageDiv);
    
    // Loading Spinner
    aiMessageDiv.innerHTML = '<span class="loading">Thinking...</span>';
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: text })
        });

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        
        let buffer = ""; 
        let aiText = "";
        let isFirstToken = true;

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            // 1. Decode the chunk and add to buffer
            const chunk = decoder.decode(value, { stream: true });
            buffer += chunk;

            // 2. Split by newline to find complete JSON objects
            const lines = buffer.split('\n');

            // 3. Keep the last part in the buffer (it might be incomplete!)
            buffer = lines.pop(); 

            // 4. Process all the COMPLETE lines
            for (const line of lines) {
                if (!line.trim()) continue;

                try {
                    const data = JSON.parse(line);

                    if (data.type === "token") {
                        if (isFirstToken) {
                            aiMessageDiv.innerHTML = "";
                            isFirstToken = false;
                        }
                        aiText += data.content;
                        // Use marked.parse to render Markdown
                        aiMessageDiv.innerHTML = marked.parse(aiText);
                        chatHistory.scrollTop = chatHistory.scrollHeight;
                    }

                    if (data.type === "sources") {
                        renderSources(data.content, aiMessageDiv);
                    }

                    if (data.error) {
                        throw new Error(data.error);
                    }

                } catch (e) {
                    console.error("Skipping invalid JSON chunk", e);
                }
            }
        }

    } catch (error) {
        aiMessageDiv.innerHTML += `<br><span style="color:red">Error: ${error.message}</span>`;
    }
}

function addMessage(content, className, isHtml = false) {
    const div = document.createElement('div');
    div.className = `message ${className}`;
    if (isHtml) div.innerHTML = content;
    else div.innerHTML = marked.parse(content);
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function renderSources(sources, containerDiv) {
    if (!sources || sources.length === 0) return;

    const uniqueSources = sources.filter((item, index, self) =>
        index === self.findIndex((t) => (t.page === item.page))
    );

    let html = `<div class="sources"><strong>Sources:</strong><br>`;
    uniqueSources.forEach((src) => {
        html += `<a class="source-link" onclick="openPdf(${src.page})">📄 Open Page ${src.page}</a>`;
    });
    html += `</div>`;
    
    containerDiv.innerHTML += html; 
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function openPdf(page) {
    modalHeader.innerText = `📄 Document Viewer (Page ${page})`;
    pdfViewer.src = `${PDF_BASE_URL}#page=${page}`;
    modal.style.display = "flex";
}

function closeModal() {
    modal.style.display = "none";
    pdfViewer.src = "";
}

window.onclick = function(event) {
    if (event.target == modal) {
        closeModal();
    }
}