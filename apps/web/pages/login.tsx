import { useEffect, useState, type FormEvent } from "react";
import Link from "next/link";
import { useRouter } from "next/router";
import { fetchMe, login } from "../lib/api";

export default function Login() {
  const router = useRouter();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchMe()
      .then(() => router.replace("/"))
      .catch(() => undefined);
  }, [router]);

  async function handleLogin(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!username.trim() || !password.trim()) {
      setStatus("Enter a username and password to continue.");
      return;
    }
    setStatus("");
    setLoading(true);
    try {
      await login(username, password);
      router.push("/");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <div className="auth-layout">
        <section className="panel auth-panel">
          <h1>Sign in</h1>
          <p className="muted">
            Use your API credentials to create runs, preflight datasets, and view history.
            Username or email-style identifiers are supported.
          </p>

          {status ? (
            <div className="status">
              <p>{status}</p>
            </div>
          ) : null}

          <form className="auth-form" onSubmit={handleLogin}>
            <div>
              <label>Username or email</label>
              <input value={username} onChange={(event) => setUsername(event.target.value)} />
            </div>
            <div>
              <label>Password</label>
              <input
                type="password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
              />
            </div>
            <div className="inline-actions">
              <button type="submit" disabled={loading}>
                {loading ? "Signing in..." : "Sign In"}
              </button>
              <Link className="link" href="/">
                Back to overview
              </Link>
            </div>
          </form>
        </section>
      </div>
      <footer className="site-footer">
        <p>
          Made by Vasco Hinostroza.
          <a href="https://github.com/theocsav">GitHub</a>
          <a href="https://www.linkedin.com/in/vasco-hinostroza/">LinkedIn</a>
          <a href="https://www.researchgate.net/profile/Vasco-Hinostroza-Fuentes">ResearchGate</a>
          <a href="mailto:vasco.hinostroza@ufl.edu">vasco.hinostroza@ufl.edu</a>
        </p>
      </footer>
    </main>
  );
}
