import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Lightbulb } from 'lucide-react';
import './ExampleQuestions.css';

function ExampleQuestions() {
  const [isExpanded, setIsExpanded] = useState(false);

  const exampleQuestions = [
    "What is the scientific name of Sokhrus?",
    "What are the medicinal uses of Sokhrus?",
    "How do I prepare Sokhrus?",
    "What are the side effects of Sokhrus?",
    "Tell me about Sokhrus herb",
    "Where can I find Sokhrus?",
    "What family does Sokhrus belong to?",
    "What parts of Sokhrus are used medicinally?",
    "Is Sokhrus safe to use?",
    "What type of plant is Sokhrus?"
  ];

  const handleQuestionClick = (question) => {
    // This will be handled by the parent component or we can emit an event
    // For now, we'll just log it
    console.log('Example question clicked:', question);
  };

  return (
    <div className="example-questions">
      <div className="example-header" onClick={() => setIsExpanded(!isExpanded)}>
        <div className="example-title">
          <Lightbulb className="example-icon" />
          <span>Example Questions</span>
        </div>
        <button className="expand-button">
          {isExpanded ? <ChevronUp className="chevron-icon" /> : <ChevronDown className="chevron-icon" />}
        </button>
      </div>
      
      {isExpanded && (
        <div className="questions-grid">
          {exampleQuestions.map((question, index) => (
            <button
              key={index}
              className="question-button"
              onClick={() => handleQuestionClick(question)}
            >
              {question}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default ExampleQuestions;
