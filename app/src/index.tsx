import ReactDOM from "react-dom/client";
import ChatComponent from "../src/Components/Chat";
import "./index.css";

const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);

root.render(
  <div className="flex-container">
    <h1>EMOTIONAI</h1>
    <div className="chat">
      <ChatComponent />
    </div>
  </div>
);
