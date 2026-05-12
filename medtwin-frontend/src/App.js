import { useState, useEffect } from "react";
import LoginPage from "./LoginPage";
import MedTwinDashboard from "./MedTwinDashboard";

export default function App() {
  const [user, setUser] = useState(null);
  const [ready, setReady] = useState(false);

  /* restore session from sessionStorage */
  useEffect(() => {
    try {
      const saved = sessionStorage.getItem("medtwin_user");
      if (saved) setUser(JSON.parse(saved));
    } catch {}
    setReady(true);
  }, []);

  const handleLogin = (userObj) => {
    sessionStorage.setItem("medtwin_user", JSON.stringify(userObj));
    setUser(userObj);
  };

  const handleLogout = () => {
    sessionStorage.removeItem("medtwin_user");
    setUser(null);
  };

  if (!ready) return null;

  if (!user) return <LoginPage onLogin={handleLogin} />;

  return <MedTwinDashboard user={user} onLogout={handleLogout} />;
}