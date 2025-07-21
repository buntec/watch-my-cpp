// import "preact/debug";
import { render } from "preact";
import { html } from "htm/preact";
import { useState, useEffect, useReducer, useRef } from "preact/hooks";
import lodash from "lodash";

// helper for notifications
// function escapeHtml(html) {
//   const div = document.createElement("div");
//   div.textContent = html;
//   return div.innerHTML;
// }

// custom function to emit shoelace toast notifications
async function notify(
  message,
  variant = "primary",
  icon = "info-circle",
  duration = 5000,
) {
  const alert = Object.assign(document.createElement("sl-alert"), {
    variant,
    closable: true,
    duration: duration,
    innerHTML: `
        <sl-icon name="${icon}" slot="icon"></sl-icon>
        ${message}
      `,
  });

  document.body.append(alert);

  await customElements.whenDefined("sl-alert");

  await alert.toast();
}

const reducer = (state, action) => {
  if (!action.type) {
    return state;
  }

  // console.log(`dispatch: ${action.type}`);

  switch (action.type) {
    case "filter_expr":
      return { ...state, filterExpr: action.filterExpr };
    case "sort_by":
      return { ...state, sortBy: action.sortBy };
    case "row_limit":
      return { ...state, rowLimit: action.rowLimit };
    case "status":
      return { ...state, status: action.status };
    case "keep_alive":
      return state;
    case "file":
      return {
        ...state,
        files: { ...state.files, [action.file.path]: action.file },
      };
    case "file_deleted": {
      let files = { ...state.files };
      delete files[action.file];
      return { ...state, files: { ...files } };
    }
    case "toggle_show_diagnostics":
      return { ...state, showDiagnostics: !state.showDiagnostics };
    case "notifications":
      return { ...state, notifications: action.notifications };
    default:
      throw new Error(`Unexpected action type: ${action.type}`);
  }
};

export function useWebSocket(url) {
  const [lastMessage, setLastMessage] = useState(null);
  const ws = useRef(null);
  const heartbeatInterval = useRef(null);

  useEffect(() => {
    function connect() {
      ws.current = new WebSocket(url);

      ws.current.onopen = () => {
        console.log("WebSocket connected");

        // Start heartbeat
        heartbeatInterval.current = setInterval(() => {
          if (ws.current && ws.current.readyState === WebSocket.OPEN) {
            ws.current.send(
              JSON.stringify({ type: "heartbeat", timestamp: Date.now() }),
            );
          }
        }, 3000);
      };

      ws.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          setLastMessage(data);
        } catch (error) {
          setLastMessage({ raw: event.data });
        }
      };

      ws.current.onerror = (err) => {
        console.error("WebSocket error", err);
        notify(
          "WebSocket error - check the server!",
          "danger",
          "exclamation-octagon",
          3000,
        );
      };

      ws.current.onclose = (event) => {
        console.log("WebSocket closed: ", event);
        notify(
          "WebSocket connection closed - attempting to reconnect...",
          "warning",
          "exclamation-triangle",
          3000,
        );
        console.log("Attempting to reconnect WebSocket...");
        setTimeout(() => connect(), 3000);
      };
    }

    connect();

    return () => {
      if (ws.current) {
        ws.current.close();
      }
      if (heartbeatInterval.current) {
        clearInterval(heartbeatInterval.current);
      }
    };
  }, [url]);

  const sendJsonMessage = (o) => {
    if (ws.current?.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify(o));
    }
  };

  return { lastMessage, sendJsonMessage };
}

const initialState = {
  files: {},
  status: {},
  sortBy: "severity",
  rowLimit: "50",
  showDiagnostics: false,
  filterExpr: "",
  notifications: [],
};

function SelectSortBy({ sortBy, setSortBy }) {
  return html`
    <sl-select
      class="select-sort-by"
      size="small"
      value=${sortBy}
      onsl-change=${(e) => setSortBy(e.target.value)}
    >
      <sl-icon name="sort-down" slot="prefix"></sl-icon>
      <sl-badge pill variant="neutral" slot="suffix">Sort by</sl-badge>
      <sl-option value="severity">Severity</sl-option>
      <sl-option value="compile_time">Compile time</sl-option>
      <sl-option value="compile_timestamp">Last compiled</sl-option>
    </sl-select>
  `;
}

function SelectRowLimit({ rowLimit, setRowLimit }) {
  return html`
    <sl-select
      class="select-row-limit"
      size="small"
      value=${rowLimit}
      onsl-change=${(e) => setRowLimit(e.target.value)}
    >
      <sl-icon name="table" slot="prefix"></sl-icon>
      <sl-badge pill variant="neutral" slot="suffix">Row limit</sl-badge>
      <sl-option value="10">10</sl-option>
      <sl-option value="25">25</sl-option>
      <sl-option value="50">50</sl-option>
      <sl-option value="100">100</sl-option>
      <sl-option value="all">All</sl-option>
    </sl-select>
  `;
}

function ToggleDiagnostics({ showDiagnostics, toggleShowDiagnostics }) {
  return html`
    <sl-tooltip content="Unfold diagnostics for all sources">
      <sl-switch
        checked=${showDiagnostics}
        onsl-change=${() => toggleShowDiagnostics()}
        size="small"
        >Expand all</sl-switch
      >
    </sl-tooltip>
  `;
}

function App() {
  const { lastMessage, sendJsonMessage } = useWebSocket(
    "ws://localhost:8000/ws",
  );
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    if (lastMessage) {
      if (Array.isArray(lastMessage)) {
        lastMessage.forEach(dispatch);
      } else {
        dispatch(lastMessage);
      }
    }
  }, [lastMessage]);

  useEffect(() => {
    state.notifications.forEach((notif) => {
      const kind = notif.kind;
      const variant =
        kind == "info"
          ? "primary"
          : kind == "warning"
            ? "warning"
            : kind == "error"
              ? "danger"
              : "neutral";

      const icon =
        kind == "info"
          ? "info-circle"
          : kind == "warning"
            ? "exclamation-triangle"
            : kind == "error"
              ? "exclamation-octagon"
              : "gear";
      notify(notif.message, variant, icon, 5000);
    });
  }, [state.notifications]);

  let files = Object.entries(state.files);

  if (state.filterExpr) {
    files = files.filter(([f, _]) =>
      f.toLocaleLowerCase().includes(state.filterExpr.toLowerCase()),
    );
  }

  const levelToInt = (level) => {
    if (level == "error") {
      return 100_000;
    }
    if (level == "warning") {
      return 1000;
    }
    if (level == "performance") {
      return 100;
    }
    if (level == "style") {
      return 10;
    }
    return 0;
  };

  const score_compile_time = (attrs) => {
    if (attrs.compile_time) {
      return attrs.compile_time;
    }
    return 0;
  };

  const score_severity = (attrs) => {
    const diags = attrs.diagnostics;

    const diag_score = diags
      ? diags.reduce((acc, value) => acc + levelToInt(value.level), 0)
      : 0;

    let status_score = 0;
    if (attrs.status == "compiling") {
      status_score = 5;
    } else if (attrs.status == "compiled") {
      status_score = 2;
    }

    return diag_score + status_score;
  };

  const sortBySeverity = ([f1, a], [f2, b]) => {
    if (f1 == f2) {
      return 0;
    }

    const sa = score_severity(a);
    const sb = score_severity(b);

    if (sa != sb) {
      return sb - sa;
    }

    if (f1 < f2) {
      return -1;
    }

    return 1;
  };

  const sortByCompileTime = ([f1, a], [f2, b]) => {
    if (f1 == f2) {
      return 0;
    }

    const sa = score_compile_time(a);
    const sb = score_compile_time(b);

    if (sa != sb) {
      return sb - sa;
    }

    if (f1 < f2) {
      return -1;
    }

    return 1;
  };

  const sortByCompileTimestamp = ([f1, a], [f2, b]) => {
    if (f1 == f2) {
      return 0;
    }

    if (a.compile_timestamp == b.compile_timestamp) {
      if (f1 < f2) {
        return -1;
      }

      return 1;
    }

    if (!a.compile_timestamp) {
      return 1;
    }

    if (!b.compile_timestamp) {
      return -1;
    }

    if (a.compile_timestamp > b.compile_timestamp) {
      return -1;
    }

    if (a.compile_timestamp < b.compile_timestamp) {
      return 1;
    }
  };

  const compareFn =
    state.sortBy == "compile_time"
      ? sortByCompileTime
      : state.sortBy == "compile_timestamp"
        ? sortByCompileTimestamp
        : sortBySeverity;

  files.sort(compareFn);

  const isTruncated =
    state.rowLimit != "all" && state.rowLimit < Object.keys(files).length;

  const topFiles =
    state.rowLimit == "all" ? files : files.slice(0, state.rowLimit);

  const Rows = topFiles.map(([file, attrs]) => {
    const [showDiagnostics, setShowDiagnostics] = useState(false);
    //

    const diags = attrs.diagnostics;

    const errors = diags
      ? diags.reduce((acc, diag) => acc + (diag.level == "error" ? 1 : 0), 0)
      : 0;

    const warnings = diags
      ? diags.reduce((acc, diag) => acc + (diag.level == "warning" ? 1 : 0), 0)
      : 0;

    const notes = diags
      ? diags.reduce((acc, diag) => acc + (diag.level == "note" ? 1 : 0), 0)
      : 0;

    const styleNotes = diags
      ? diags.reduce((acc, diag) => acc + (diag.level == "style" ? 1 : 0), 0)
      : 0;

    const perfNotes = diags
      ? diags.reduce(
          (acc, diag) => acc + (diag.level == "performance" ? 1 : 0),
          0,
        )
      : 0;

    const StatusIcons =
      attrs.status == "compiling"
        ? html`<div class="status-symbol"><sl-spinner></sl-spinner></div>`
        : attrs.status == "new"
          ? html`<div class="status-symbol">
              <sl-icon name="question"></sl-icon>
            </div>`
          : errors > 0 || warnings > 0 || styleNotes > 0 || perfNotes > 0
            ? html`<div class="status-symbols">
                <div class="error">
                  ${errors > 0 &&
                  html` <sl-icon name="x-octagon"></sl-icon>(${errors}) `}
                </div>
                <div class="warning">
                  ${warnings > 0 &&
                  html`
                    <sl-icon name="exclamation-triangle"></sl-icon>(${warnings})
                  `}
                </div>
                <div class="performance">
                  ${perfNotes > 0 &&
                  html` <sl-icon name="speedometer"></sl-icon>(${perfNotes}) `}
                </div>
                <div class="style">
                  ${styleNotes > 0 &&
                  html`
                    <sl-icon name="emoji-sunglasses"></sl-icon>(${styleNotes})
                  `}
                </div>
              </div>`
            : html`<div class="status-symbol">
                <sl-icon name="check2-circle"></sl-icon>
              </div>`;

    const statusClass =
      attrs.status == "compiling"
        ? "compiling"
        : errors > 0
          ? "error"
          : warnings > 0
            ? "warning"
            : notes > 0
              ? "info"
              : "ok";

    const ButtonRecompile = html`
      <sl-tooltip content="Recompile">
        <sl-button
          size="small"
          outline
          onClick=${(e) => {
            e.stopPropagation();
            sendJsonMessage({ type: "recompile_file", file: file });
          }}
          >🛠️</sl-button
        ></sl-tooltip
      >
    `;

    const ButtonRecompileForceClangTidy = html`
      <sl-tooltip content="Clang-tidy">
        <sl-button
          size="small"
          outline
          onClick=${(e) => {
            e.stopPropagation();
            sendJsonMessage({
              type: "recompile_file_and_force_clang_tidy",
              file: file,
            });
          }}
          >🧹</sl-button
        ></sl-tooltip
      >
    `;

    const ButtonRecompileForceIWYU = html`
      <sl-tooltip content="include-what-you-use">
        <sl-button
          size="small"
          outline
          onClick=${(e) => {
            e.stopPropagation();
            sendJsonMessage({
              type: "recompile_file_and_force_iwyu",
              file: file,
            });
          }}
          >🤖</sl-button
        ></sl-tooltip
      >
    `;

    const ButtonRecompileForceCppCheck = html`
      <sl-tooltip content="cppcheck">
        <sl-button
          size="small"
          outline
          onClick=${(e) => {
            e.stopPropagation();
            sendJsonMessage({
              type: "recompile_file_and_force_cppcheck",
              file: file,
            });
          }}
          >✔️</sl-button
        ></sl-tooltip
      >
    `;

    const ButtonToClipboard = html`<sl-copy-button
      value=${file}
    ></sl-copy-button>`;

    const UserActions = html` <div class="user-actions">
      ${ButtonToClipboard} ${ButtonRecompile} ${ButtonRecompileForceClangTidy}
      ${ButtonRecompileForceCppCheck} ${ButtonRecompileForceIWYU}
    </div>`;

    const unfoldDiags =
      diags && diags.length > 0 && (state.showDiagnostics || showDiagnostics);

    const SourceWithDiagnostics = html`<div class="source-with-diagnostics">
      <div class="source-with-buttons">
        <div class="chevron ${unfoldDiags ? "rot-90" : null}">
          <sl-icon name="chevron-right"></sl-icon>
        </div>
        <div class="sourcepath ${statusClass}">${file}</div>
      </div>
      ${unfoldDiags &&
      html` <div class="diagnostics">
        <div class="diagnostics-header">
          <div>Line</div>
          <div>Col</div>
          <div>Source</div>
          <div>Kind</div>
          <div>File</div>
          <div>Message</div>
          <div>Category</div>
        </div>
        ${diags.map(
          (diag) =>
            html`<div class="diagnostic ${diag.level}">
              <div class="line-number">${diag.line}</div>
              <div class="col-number">${diag.column}</div>
              <div class="diagnostic-source">${diag.source}</div>
              <div>${diag.level}</div>
              <div class="diagnostic-source-file">
                ${diag.file != file ? diag.file_short : '"'}
              </div>
              <div class="diagnostic-message">${diag.message}</div>
              <div class="diagnostic-categories">
                ${diag.categories?.length > 0 ? `[${diag.categories}]` : ""}
              </div>
            </div>`,
        )}
      </div>`}
    </div>`;

    return html`
      <div class="row" key=${file}>
        <div
          class="cell filename"
          onClick=${() => setShowDiagnostics(!showDiagnostics)}
        >
          ${SourceWithDiagnostics}
        </div>
        <div class="cell actions">${UserActions}</div>
        <div class="cell status ${statusClass}">${StatusIcons}</div>
        <div class="cell compile-time">${attrs.compile_time?.toFixed(2)}</div>
        <div class="cell compile-timestamp">${attrs.compile_timestamp}</div>
      </div>
    `;
  });

  const Header = html`
    <div class="row header">
      <div class="header-cell">Source</div>
      <div class="header-cell" id="header-actions">Actions</div>
      <div class="header-cell" id="header-status">Status</div>
      <div class="header-cell stopwatch">
        <sl-icon name="stopwatch"></sl-icon>
      </div>
      <div class="header-cell">Last compiled</div>
    </div>
  `;

  const Ellipsis = html`
    <div class="row ellipsis">
      <div>Increase row limit to see more...</div>
      <div></div>
      <div></div>
      <div></div>
      <div></div>
    </div>
  `;

  const Logo = html`<div id="logo">👀</div>`;

  const maxWorkers = state.status?.settings?.max_workers;
  const workers = state.status?.settings?.workers;

  const SelectWorkers =
    workers &&
    maxWorkers &&
    html`
      <sl-select
        id="select-workers"
        value=${`${workers}`}
        size="small"
        onsl-change=${(e) =>
          sendJsonMessage({
            type: "num_workers_change",
            n: parseInt(e.target.value),
          })}
      >
        <sl-badge pill variant="neutral" slot="suffix"># Workers</sl-badge>
        ${lodash
          .range(1, maxWorkers + 1)
          .map((i) => html`<sl-option value=${i}>${i}</sl-option>`)}
      </sl-select>
    `;

  const ButtonPause = html`
    <sl-tooltip content="Pause compilation">
      <sl-icon-button
        disabled=${state.status?.paused ? true : null}
        onClick=${(e) => {
          e.stopPropagation();
          sendJsonMessage({ type: "pause" });
        }}
        name="pause"
      ></sl-icon-button
    ></sl-tooltip>
  `;

  const ButtonResume = html`
    <sl-tooltip content="Resume compilation">
      <sl-icon-button
        disabled=${state.status?.paused == false ? true : null}
        onClick=${(e) => {
          e.stopPropagation();
          sendJsonMessage({ type: "resume" });
        }}
        name="play"
      ></sl-icon-button
    ></sl-tooltip>
  `;

  const InputFilter = html`
    <sl-input
      clearable
      value=${state.filterExpr}
      onsl-input=${(ev) =>
        dispatch({ type: "filter_expr", filterExpr: ev.target.value })}
      placeholder="Filter"
      size="small"
    >
      <sl-icon name="funnel" slot="prefix"></sl-icon>
    </sl-input>
  `;

  const KillSwitch = html`<sl-button
    size="small"
    variant="danger"
    onClick=${() => sendJsonMessage({ type: "kill_switch" })}
    >Restart</sl-button
  >`;

  const status = state.status;

  const pctDone =
    status && state.status.n_sources
      ? 100 -
        (100.0 * (status.n_queued + status.n_queued_low_prio)) /
          status.n_sources
      : 100;

  const busy = pctDone != 100;

  const ProgressBar = html` <sl-progress-bar
    id="progress-bar"
    class=${busy ? "busy" : null}
    value="${pctDone}"
  ></sl-progress-bar>`;

  const ToggleClangTidy = html`
    <sl-tooltip
      content="Warning: enabling clang-tidy may slow down compilation dramatically!"
    >
      <sl-switch
        disabled=${state.status?.settings ? null : true}
        checked=${state.status?.settings?.clang_tidy}
        onsl-change=${() => sendJsonMessage({ type: "toggle_clang_tidy" })}
        size="small"
        >Clang-tidy
      </sl-switch>
    </sl-tooltip>
  `;

  const ToggleIWYU = html`
    <sl-tooltip
      content="Warning: enabling include-what-you-use may slow down compilation dramatically!"
    >
      <sl-switch
        disabled=${state.status?.settings ? null : true}
        checked=${state.status?.settings?.iwyu}
        onsl-change=${() => sendJsonMessage({ type: "toggle_iwyu" })}
        size="small"
        >IWYU
      </sl-switch>
    </sl-tooltip>
  `;

  const ToggleCppCheck = html`
    <sl-tooltip
      content="Warning: enabling cppcheck may slow down compilation dramatically!"
    >
      <sl-switch
        disabled=${state.status?.settings ? null : true}
        checked=${state.status?.settings?.cppcheck}
        onsl-change=${() => sendJsonMessage({ type: "toggle_cppcheck" })}
        size="small"
        >Cppcheck
      </sl-switch>
    </sl-tooltip>
  `;

  const CompileCommandsInfo =
    state.status?.settings?.compile_commands_path &&
    html`
      <div id="compile-commands-info">
        ${state.status?.settings?.compile_commands_path}
      </div>
    `;

  const ExcludePatternsInfo =
    state.status?.settings?.ignore_patterns &&
    html`
      <div id="exlude-patterns-info">
        <span>exclude: </span>
        <span class="mono">${state.status?.settings?.ignore_patterns}</span>
      </div>
    `;

  const IncludePatternsInfo =
    state.status?.settings?.include_patterns &&
    html`
      <div id="include-patterns-info">
        <span>include: </span
        ><span class="mono">${state.status?.settings?.include_patterns}</span>
      </div>
    `;

  const GridMain =
    topFiles &&
    topFiles.length > 0 &&
    html` <div class="files-grid">
      ${Header} ${Rows} ${isTruncated ? Ellipsis : null}
    </div>`;

  const Controls = html`
    <div id="controls">${ButtonPause} ${ButtonResume}</div>
  `;

  return html`
    <div class="app">
      ${ProgressBar}
      <div class="ribbon">
        ${Logo}
        <${SelectSortBy}
          sortBy=${state.sortBy}
          setSortBy=${(sortBy) => dispatch({ type: "sort_by", sortBy })}
        />
        ${InputFilter}
        <${SelectRowLimit}
          rowLimit=${state.rowLimit}
          setRowLimit=${(rowLimit) => dispatch({ type: "row_limit", rowLimit })}
        />
        ${SelectWorkers}
        <${ToggleDiagnostics}
          showDiagnostics=${state.showDiagnostics}
          toggleShowDiagnostics=${() =>
            dispatch({ type: "toggle_show_diagnostics" })}
        />
        ${ToggleClangTidy} ${ToggleCppCheck} ${ToggleIWYU} ${Controls}
        ${KillSwitch} ${CompileCommandsInfo} ${ExcludePatternsInfo}
        ${IncludePatternsInfo}
      </div>
      ${GridMain}
    </div>
  `;
}

render(html`<${App} />`, document.body);
