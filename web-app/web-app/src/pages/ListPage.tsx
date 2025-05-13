import { useEffect, useState } from "react";
import { getProblems } from "../api/problems";
import type { Problem } from "../api/problems";
import Input from "../components/Input";
import { colors, spacing } from "../theme";

export default function ListPage() {
  const [items, setItems] = useState<Problem[]>([]);
  const [klass, setKlass] = useState("");
  const [type, setType] = useState<"all" | "image" | "video">("all");

  useEffect(() => {
    getProblems({
      media_type: type === "all" ? undefined : type,
    }).then(items => {
      const filtered = klass 
        ? items.filter(p => 
            p.predicted_classes.some(c => 
              c.toLowerCase().includes(klass.toLowerCase())
            )
          )
        : items;
      setItems(filtered);
    });
  }, [klass, type]);

  return (
    <div style={{ padding: spacing.m }}>
      <h2>Problems list</h2>

      <div style={{ display: "flex", gap: spacing.s }}>
        <Input
          placeholder="filter by class…"
          value={klass}
          onChange={(e) => setKlass(e.target.value)}
        />

        <select
          value={type}
          onChange={(e) => setType(e.target.value as "all" | "image" | "video")}
        >
          <option value="all">all types</option>
          <option value="image">images</option>
          <option value="video">videos</option>
        </select>
      </div>

      <table
        style={{
          width: "100%",
          marginTop: spacing.m,
          background: colors.surface,
          borderCollapse: "collapse",
        }}
      >
        <thead>
          <tr>
            <th>ID</th>
            <th>Date</th>
            <th>User</th>
            <th>Classes</th>
            <th>Thumb</th>
          </tr>
        </thead>
        <tbody>
          {items.map((it) => (
            <tr key={it.media_id}>
              <td>{it.media_id}</td>
              <td>{new Date(it.created_at).toLocaleString()}</td>
              <td>{it.user_username}</td>
              <td>{it.predicted_classes.join(", ")}</td>
              <td>
                {it.annotated_image_url && (
                  <img
                    src={import.meta.env.VITE_API_URL + it.annotated_image_url}
                    style={{ width: 80 }}
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