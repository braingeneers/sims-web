import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";

// Render the top-level <App> component into the #root div
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);