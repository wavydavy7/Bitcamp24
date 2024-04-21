import React from "react";
import ReactDOM from "react-dom/client";
import ChatComponent from "../src/Components/Chat";
import Video from "../src/Components/Video";
import "./app.css";
const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);

root.render(
  <div>
    <div className="chat">
      <ChatComponent />
    </div>

    <div className="vid">
      <Video />
    </div>
  </div>
);
