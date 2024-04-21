import './App.css';
import {useState, useEffect} from 'react'
import { motion } from "framer-motion";

function App() {

  const [accuracy, setAccuracy] = useState(5)

  useEffect(() => {
    fetch("/api/ml").then(res => res.json()).then(data => {setAccuracy(data.accuracy)})
  }, []);

  return (
    <div className="App bg-gray-20">
      <header className="App-header">
      <motion.div
            className="md:my-5 md:w-3/5"
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.5 }}
            transition={{ duration: 1 }}
            variants={{
              hidden: { opacity: 0, x: 50 },
              visible: { opacity: 1, x: 0 },
            }}
          >
            <p className={`my-5`}>
              A list of projects I've done.
            </p>
          </motion.div>
        <p>

        </p>
      </header>
    </div>
  );
}

export default App;
