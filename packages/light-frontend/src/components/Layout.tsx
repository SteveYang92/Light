import { Link, useLocation } from "react-router-dom";

export default function Layout({ children }: { children: React.ReactNode }) {
  const loc = useLocation();

  return (
    <div className="min-h-screen flex flex-col">
      <header className="flex items-center justify-between px-4 sm:px-8 h-14 border-b border-[#1f1f1f] shrink-0">
        <Link to="/" className="text-lg font-semibold tracking-tight">
          Light
        </Link>
        <nav className="flex items-center gap-4">
          <Link
            to="/"
            className={`text-sm ${loc.pathname === "/" ? "text-[#e5e5e5]" : "text-[#6b7280] hover:text-[#e5e5e5]"}`}
          >
            视频库
          </Link>
          <Link
            to="/download"
            className={`text-sm ${loc.pathname === "/download" ? "text-[#e5e5e5]" : "text-[#6b7280] hover:text-[#e5e5e5]"}`}
          >
            下载
          </Link>
          <Link
            to="/import"
            className={`text-sm ${loc.pathname === "/import" ? "text-[#e5e5e5]" : "text-[#6b7280] hover:text-[#e5e5e5]"}`}
          >
            导入
          </Link>
        </nav>
      </header>
      <main className="flex-1 px-4 sm:px-8 py-6 max-w-7xl w-full mx-auto">
        {children}
      </main>
    </div>
  );
}
