import { Routes, Route, Navigate } from "react-router-dom";
import Layout from "./components/Layout";
import Download from "./pages/Download";
import Import from "./pages/Import";
import Library from "./pages/Library";
import Player from "./pages/Player";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Library />} />
        <Route path="/download" element={<Download />} />
        <Route path="/import" element={<Import />} />
        <Route path="/new" element={<Navigate to="/download" replace />} />
        <Route path="/watch/:id" element={<Player />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}
