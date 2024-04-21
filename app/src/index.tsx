import ReactDOM from "react-dom/client";
import ChatComponent from "../src/Components/Chat";
import Video from "../src/Components/Video";
import App from "./App";
const root = ReactDOM.createRoot(
  document.getElementById("root") as HTMLElement
);

root.render(
  <div>
    <div className="chat">
      <ChatComponent />
    </div>
    <div>
      <App />
    </div>
    
    <div className="vid">
      <Video />
    </div>
  </div>
);
