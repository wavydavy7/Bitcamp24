import './App.css';
import {useState, useEffect} from 'react'

function App() {

  const [accuracy, setAccuracy] = useState(5)

  useEffect(() => {
    fetch("/api/ml").then(res => res.json()).then(data => {setAccuracy(data.accuracy)})
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>
          Output: {accuracy}
        </p>
      </header>
    </div>
  );
}

export default App;
