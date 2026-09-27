import http from "node:http";
import { URL } from "node:url";

const LISTEN_HOST = process.env.KIRO_SHIM_HOST || "127.0.0.1";
const LISTEN_PORT = Number(process.env.KIRO_SHIM_PORT || "8766");
const UPSTREAM = new URL(process.env.KIRO_UPSTREAM || "http://127.0.0.1:8765");
const MAX_TOOL_TEXT_CHARS = Number(process.env.KIRO_SHIM_MAX_TOOL_TEXT_CHARS || "6000");

const TOOL_NAME_MAP = new Map([
  ["agent", "Agent"],
  ["askUserQuestion", "AskUserQuestion"],
  ["bash", "Bash"],
  ["cronCreate", "CronCreate"],
  ["cronDelete", "CronDelete"],
  ["cronList", "CronList"],
  ["edit", "Edit"],
  ["enterPlanMode", "EnterPlanMode"],
  ["enterWorktree", "EnterWorktree"],
  ["exitPlanMode", "ExitPlanMode"],
  ["exitWorktree", "ExitWorktree"],
  ["glob", "Glob"],
  ["grep", "Grep"],
  ["notebookEdit", "NotebookEdit"],
  ["read", "Read"],
  ["scheduleWakeup", "ScheduleWakeup"],
  ["skill", "Skill"],
  ["taskOutput", "TaskOutput"],
  ["taskStop", "TaskStop"],
  ["todoWrite", "TodoWrite"],
  ["webFetch", "WebFetch"],
  ["webSearch", "WebSearch"],
  ["write", "Write"],
]);

function rewriteToolNames(text) {
  return text.replace(/"name"\s*:\s*"([^"]+)"/g, (match, name) => {
    const mapped = TOOL_NAME_MAP.get(name);
    return mapped ? match.replace(`"${name}"`, `"${mapped}"`) : match;
  });
}

function createStreamingRewriter(write) {
  let carry = "";

  return {
    write(chunk) {
      const text = carry + chunk;
      const splitAt = Math.max(0, text.length - 256);
      carry = text.slice(splitAt);
      if (splitAt > 0) write(rewriteToolNames(text.slice(0, splitAt)));
    },
    end() {
      if (carry) write(rewriteToolNames(carry));
      carry = "";
    },
  };
}

function truncateText(value) {
  if (typeof value !== "string" || value.length <= MAX_TOOL_TEXT_CHARS) return value;
  return `${value.slice(0, MAX_TOOL_TEXT_CHARS)}\n\n[kiro-shim: truncated ${value.length - MAX_TOOL_TEXT_CHARS} chars from oversized tool result]`;
}

function rewriteRequestBody(rawBody, contentType) {
  if (!contentType?.includes("application/json") || !rawBody) return rawBody;

  let body;
  try {
    body = JSON.parse(rawBody);
  } catch {
    return rawBody;
  }

  let changed = false;
  const visit = (value, inToolResult = false) => {
    if (!value || typeof value !== "object") return;

    if (Array.isArray(value)) {
      for (const item of value) visit(item, inToolResult);
      return;
    }

    const isToolResult =
      inToolResult ||
      value.type === "tool_result" ||
      typeof value.toolUseId === "string" ||
      typeof value.tool_use_id === "string";

    if (isToolResult) {
      if (typeof value.content === "string") {
        const next = truncateText(value.content);
        if (next !== value.content) {
          value.content = next;
          changed = true;
        }
      }

      if (Array.isArray(value.content)) {
        for (const part of value.content) {
          if (part && typeof part === "object" && typeof part.text === "string") {
            const next = truncateText(part.text);
            if (next !== part.text) {
              part.text = next;
              changed = true;
            }
          }
        }
      }
    }

    for (const child of Object.values(value)) visit(child, isToolResult);
  };

  visit(body);
  return changed ? JSON.stringify(body) : rawBody;
}

const server = http.createServer((clientReq, clientRes) => {
  const chunks = [];
  clientReq.on("data", (chunk) => chunks.push(chunk));
  clientReq.on("end", () => {
    const rawBody = Buffer.concat(chunks).toString("utf8");
    const requestBody = rewriteRequestBody(rawBody, clientReq.headers["content-type"]);
    const headers = {
      ...clientReq.headers,
      host: UPSTREAM.host,
      "content-length": Buffer.byteLength(requestBody),
    };

    const upstreamReq = http.request(
      {
        protocol: UPSTREAM.protocol,
        hostname: UPSTREAM.hostname,
        port: UPSTREAM.port,
        method: clientReq.method,
        path: clientReq.url,
        headers,
      },
      (upstreamRes) => {
        clientRes.writeHead(upstreamRes.statusCode || 502, upstreamRes.headers);
        upstreamRes.setEncoding("utf8");
        const rewriter = createStreamingRewriter((chunk) => clientRes.write(chunk));
        upstreamRes.on("data", (chunk) => rewriter.write(chunk));
        upstreamRes.on("end", () => {
          rewriter.end();
          clientRes.end();
        });
      },
    );

    upstreamReq.on("error", (error) => {
      clientRes.writeHead(502, { "content-type": "application/json" });
      clientRes.end(JSON.stringify({ error: `kiro shim upstream error: ${error.message}` }));
    });

    upstreamReq.end(requestBody);
  });
});

server.listen(LISTEN_PORT, LISTEN_HOST, () => {
  console.error(`kiro tool-name shim listening on http://${LISTEN_HOST}:${LISTEN_PORT}`);
  console.error(`forwarding to ${UPSTREAM.href}`);
});
