import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [messages, setMessages] = useState([
    {
      type: 'bot',
      content: 'Hello! I\'m your Herbal Medicine Assistant. Ask me about any herb, its uses, preparation methods, or side effects. For example, you can ask "What are the uses of Sokhrus?" or "How do you prepare Chamomile?"',
      timestamp: new Date()
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modelInfo, setModelInfo] = useState(null);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    // Fetch model info on component mount
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const response = await axios.get(`${API_BASE_URL}/model-info`);
      setModelInfo(response.data);
    } catch (error) {
      console.error('Error fetching model info:', error);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      content: inputMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    setIsLoading(true);

    try {
      const response = await axios.post(`${API_BASE_URL}/chat`, {
        message: inputMessage,
        max_length: 200,
        temperature: 0.7
      });

      const botMessage = {
        type: 'bot',
        content: response.data.response,
        confidence: response.data.confidence,
        responseTime: response.data.response_time,
        timestamp: new Date()
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      const errorMessage = {
        type: 'bot',
        content: 'I apologize, but I encountered an error while processing your question. Please make sure the API server is running and try again.',
        timestamp: new Date(),
        isError: true
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const sampleQuestions = [
    "What are the uses of Sokhrus?",
    "How do you prepare Chamomile?",
    "What are the side effects of Astragalus?",
    "Tell me about Rhodiola imbricata",
    "Which parts of Equisetum are used medicinally?"
  ];

  return (
    <div className="App">
      <header className="app-header">
        <h1>🌿 Herbal Medicine Chatbot</h1>
        <p>Your AI assistant for traditional herbal medicine information</p>
        {modelInfo && (
          <div className="model-info">
            <span>Model: {modelInfo.model_info?.model_name || 'Unknown'}</span>
            <span>•</span>
            <span>Parameters: {modelInfo.parameters?.toLocaleString() || 'N/A'}</span>
          </div>
        )}
      </header>

      <main className="chat-container">
        <div className="messages-container">
          {messages.map((message, index) => (
            <div key={index} className={`message ${message.type}`}>
              <div className="message-content">
                <div className="message-text">
                  {message.content}
                </div>
                <div className="message-meta">
                  <span className="timestamp">{formatTimestamp(message.timestamp)}</span>
                  {message.confidence && (
                    <span className="confidence">
                      Confidence: {(message.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                  {message.responseTime && (
                    <span className="response-time">
                      {message.responseTime.toFixed(2)}s
                    </span>
                  )}
                </div>
              </div>
            </div>
          ))}
          
          {isLoading && (
            <div className="message bot loading">
              <div className="message-content">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <div className="sample-questions">
            <p>Try asking:</p>
            <div className="sample-buttons">
              {sampleQuestions.map((question, index) => (
                <button
                  key={index}
                  className="sample-button"
                  onClick={() => setInputMessage(question)}
                  disabled={isLoading}
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
          
          <div className="input-row">
            <textarea
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me about any herb... (e.g., 'What are the uses of Sokhrus?')"
              disabled={isLoading}
              rows="2"
            />
            <button 
              onClick={sendMessage} 
              disabled={isLoading || !inputMessage.trim()}
              className="send-button"
            >
              {isLoading ? '⏳' : '📤'}
            </button>
          </div>
        </div>
      </main>

      <footer className="app-footer">
        <p>⚠️ This chatbot provides educational information only. Always consult healthcare professionals before using any herbal remedies.</p>
      </footer>
    </div>
  );
}

export default App;