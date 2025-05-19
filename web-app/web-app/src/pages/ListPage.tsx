import { useEffect, useState } from "react";
import { getProblems } from "../api/problems";
import type { Problem } from "../api/problems";
import Input from "../components/Input";
import styles from "../styles/ListPage.module.css";

export default function ListPage() {
  const [items, setItems] = useState<Problem[]>([]);
  const [klass, setKlass] = useState("");
  const [type, setType] = useState<"all" | "image" | "video">("all");

  useEffect(() => {
    getProblems({ media_type: type === "all" ? undefined : type }).then(
      (items) => {
        const filtered = klass
          ? items.filter((p) =>
              p.predicted_classes.some((c) =>
                c.toLowerCase().includes(klass.toLowerCase())
              )
            )
          : items;
        setItems(filtered);
      }
    );
  }, [klass, type]);

  return (
    <div style={{ padding: "var(--spacing-m)" }}>
      <h2>Problems list</h2>
      <div style={{ display: "flex", gap: "var(--spacing-s)" }}>
        <Input
          placeholder="filter by class…"
          value={klass}
          onChange={(e) => setKlass(e.target.value)}
        />
        <select
          value={type}
          onChange={(e) =>
            setType(e.target.value as "all" | "image" | "video")
          }
        >
          <option value="all">all types</option>
          <option value="image">images</option>
          <option value="video">videos</option>
        </select>
      </div>

      <table className={styles.table}>
        <thead>
          <tr>
            {["ID", "Date", "User", "Classes", "Description", "Solution", "Thumb"].map(
              (label) => (
                <th key={label} className={styles.th}>
                  {label}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody className={styles.tbody}>
          {items.map((it) => (
            <tr key={it.media_id}>
              <td className={styles.td}>{it.media_id}</td>
              <td className={styles.td}>
                {new Date(it.created_at).toLocaleString()}
              </td>
              <td className={styles.td}>{it.user_username}</td>
              <td className={styles.td}>{it.predicted_classes.join(", ")}</td>
              <td className={`${styles.td} ${styles["td--left"]}`}>
                {it.descriptions?.[0] ?? "-"}
              </td>
              <td className={`${styles.td} ${styles["td--left"]}`}>
                {it.solutions?.[0] ?? "-"}
              </td>
              <td className={styles.td}>
                {it.annotated_image_url && (
                  <img
                    src={`${import.meta.env.VITE_API_BASE}/static/${it.media_id}.jpg`}
                    style={{ width: 80, borderRadius: 4 }}
                  />
                )}
                {it.annotated_video_url && "🎞️"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
