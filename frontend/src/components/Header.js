import React from 'react';
import { Bot, Wifi, WifiOff } from 'lucide-react';
import './Header.css';

function Header({ isConnected, connectionStatus, onReconnect }) {
  return (
    <header className="header">
      <div className="header-content">
        <div className="logo">
          <Bot className="logo-icon" />
          <h1>Herbal Medicine Chatbot</h1>
        </div>
        
        <div className="connection-status">
          <div className={`status-indicator ${isConnected ? 'connected' : 'disconnected'}`}>
            {isConnected ? <Wifi className="status-icon" /> : <WifiOff className="status-icon" />}
            <span className="status-text">{connectionStatus}</span>
          </div>
          
          {!isConnected && (
            <button onClick={onReconnect} className="reconnect-button">
              Reconnect
            </button>
          )}
        </div>
      </div>
    </header>
  );
}

export default Header;
