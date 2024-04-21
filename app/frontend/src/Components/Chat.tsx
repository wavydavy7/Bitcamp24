import React, { useState } from "react";
import { fetchChatGPTResponse } from "../api";
import "./chat.css";
import { render } from "react-dom";
const ChatComponent = () => {
  console.log("success");
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");

  const handleInputChange = (event: {
    target: { value: React.SetStateAction<string> };
  }) => {
    setPrompt(event.target.value);
  };

  const handleSubmit = async (event: { preventDefault: () => void }) => {
    event.preventDefault();
    if (!prompt) return;

    renderMessageEle(prompt, "user");
    setPrompt("");
    try {
      const data = await fetchChatGPTResponse(prompt);
      if (data && data.choices && data.choices.length > 0) {
        const responseText = data.choices[0].message.content;
        setResponse(responseText); // Update response state
        renderMessageEle(responseText, "chatbot"); // Render bot response
        setScrollPosition(); // Adjust scroll if necessary
      } else {
        console.error("No data received or data format incorrect.");
      }
    } catch (error) {
      console.error("Failed to fetch response:", error);
      renderMessageEle("Failed to fetch response.", "chatbot"); // Show error message in chat
    }
  };

  const renderMessageEle = (txt: string, type: string) => {
    const chatBody = document.getElementById("chat-body");
    if (!chatBody) {
      console.error("Chat body element not found!");
      return; // Exit the function if chatBody is null
    }
    let className = "user-message";
    if (type !== "user") {
      className = "chatbot-message";
    }
    const messageEle = document.createElement("div");
    const txtNode = document.createTextNode(txt);
    messageEle.classList.add(className);
    messageEle.append(txtNode);
    chatBody.append(messageEle);
  };

  const setScrollPosition = () => {
    const chatBody = document.getElementById("chat-body");
    if (!chatBody) {
      console.error("Chat body element not found!");
      return; // Exit the function if chatBody is null
    }
    if (chatBody.scrollHeight > 0) {
      chatBody.scrollTop = chatBody.scrollHeight;
    }
  };

  return (
    <div>
      <div className="container">
        <div className="chat-body" id="chat-body"></div>
        <form onSubmit={handleSubmit}>
          <div className="form-container">
            <textarea
              className="in"
              // type="text"
              value={prompt}
              onChange={handleInputChange}
            />
            <button className="sub" type="submit">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                viewBox="0 0 2048 2048"
                width="30"
                height="28"
                fill="#3485F7"
              >
                <path d="M221 1027h931l128-64-128-64H223L18 77l1979 883L18 1843l203-816z" />
              </svg>
            </button>
          </div>
        </form>
        {/* {response && <p>Response: {response}</p>} */}
      </div>
    </div>
  );
};

export default ChatComponent;
