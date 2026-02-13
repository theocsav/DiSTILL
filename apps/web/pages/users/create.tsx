import Link from "next/link";
import { useRouter } from "next/router";
import { FormEvent, useEffect, useState } from "react";
import { createUser, fetchMe } from "../../lib/api";

export default function CreateUserPage() {
  const router = useRouter();
  const [currentUser, setCurrentUser] = useState<string | null>(null);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchMe()
      .then((data) => setCurrentUser(data.username))
      .catch(() => {
        setCurrentUser(null);
        router.replace("/login");
      });
  }, [router]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!username.trim() || !password.trim()) {
      setStatus("Username and password are required.");
      return;
    }
    if (password !== confirmPassword) {
      setStatus("Passwords do not match.");
      return;
    }
    setStatus("");
    setLoading(true);
    try {
      const created = await createUser({ username, password });
      setStatus(`Created user ${created.username}.`);
      setUsername("");
      setPassword("");
      setConfirmPassword("");
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
          <h1>Create User</h1>
          <p className="muted">
            Signed in as {currentUser || "..."}. Create a new account for app access.
          </p>

          {status ? (
            <div className="status">
              <p>{status}</p>
            </div>
          ) : null}

          <form className="auth-form" onSubmit={handleSubmit}>
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
            <div>
              <label>Confirm Password</label>
              <input
                type="password"
                value={confirmPassword}
                onChange={(event) => setConfirmPassword(event.target.value)}
              />
            </div>
            <div className="inline-actions">
              <button type="submit" disabled={loading || !currentUser}>
                {loading ? "Creating..." : "Create User"}
              </button>
              <Link className="link" href="/">
                Back to overview
              </Link>
            </div>
          </form>
        </section>
      </div>
    </main>
  );
}
