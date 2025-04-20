// thoughtmap_viz.js

let svg, simulation, nodes = [], links = [];
const padding = 40;
const width = 600;
const height = 400;

const groupColors = d3.scaleOrdinal(d3.schemeTableau10);

const themeCenters = {
  identity: [150, 100],
  values: [450, 100],
  agency: [150, 300],
  desires: [450, 300],
  fulfillment: [300, 200],
  relationships: [300, 350],
  default: [width / 2, height / 2]
};

function initThoughtMapViz(containerId = "#thoughtmap") {
  const container = document.querySelector(containerId);
  if (!container) return;

  svg = d3.select(containerId)
    .append("svg")
    .attr("width", width)
    .attr("height", height);

  svg.append("g")
    .attr("id", "graph-group")
    .attr("transform", "translate(0,0)");

  simulation = d3.forceSimulation()
    .force("link", d3.forceLink().id(d => d.id).distance(100))
    .force("charge", d3.forceManyBody().strength(-80))
    .force("collision", d3.forceCollide().radius(d => d.isTension ? 8 : 12))
    .force("x", d3.forceX(d => (themeCenters[d.group] || themeCenters.default)[0]).strength(0.2))
    .force("y", d3.forceY(d => (themeCenters[d.group] || themeCenters.default)[1]).strength(0.2));

  svg.call(
    d3.zoom()
      .scaleExtent([0.1, 4])
      .on("zoom", (event) => {
        svg.select("#graph-group").attr("transform", event.transform);
      })
  );
}

function updateThoughtMap(mapDelta, resolvedIds = []) {

  const oldPositions = {};
  nodes.forEach(n => {
    oldPositions[n.id] = { x: n.x, y: n.y };
  });

  const newNodes = [];
  const newLinks = [];
  const nodeMap = {};

  for (const theme in mapDelta) {
    const group = theme;
    const beliefs = mapDelta[theme].beliefs || [];
    const tensions = mapDelta[theme].tensions || [];

    beliefs.forEach((b, i) => {
      const id = `b-${group}-${i}`;
      const existing = oldPositions[id];
      const node = {
        id,
        label: typeof b === "object" ? b.summary || JSON.stringify(b) : b,
        group,
        isTension: false,
        x: existing?.x ?? padding + Math.random() * (width - 2 * padding),
        y: existing?.y ?? padding + Math.random() * (height - 2 * padding)
      };
      newNodes.push(node);
      nodeMap[id] = node;
    });

    tensions.forEach((t, i) => {
      const id = `t-${group}-${i}`;
      const existing = oldPositions[id];
      const node = {
        id,
        label: typeof t === "object" ? t.summary || JSON.stringify(t) : t,
        group,
        isTension: true,
        x: existing?.x ?? padding + Math.random() * (width - 2 * padding),
        y: existing?.y ?? padding + Math.random() * (height - 2 * padding)
      };
      newNodes.push(node);
      nodeMap[id] = node;

      // Link each tension to ALL beliefs in the same group
      beliefs.forEach((_, j) => {
        const beliefId = `b-${group}-${j}`;
        if (nodeMap[beliefId]) {
          newLinks.push({ source: nodeMap[beliefId], target: node });
        }
      });
    });
  }

  nodes = newNodes;
  links = newLinks;

  const g = svg.select("#graph-group");

  const link = g.selectAll(".link")
    .data(links, d => `${d.source.id}-${d.target.id}`)
    .join("line")
    .attr("class", "link")
    .attr("stroke", "#888")
    .attr("stroke-width", 1.5);

  const node = g.selectAll(".node")
    .data(nodes, d => d.id)
    .join("circle")
    .attr("class", "node")
    .attr("r", d => d.isTension ? 6 : 10)
    .attr("fill", d => d.isTension ? "#f66" : groupColors(d.group))
    .on("click", (event, d) => {
      alert(`🧠 ${d.isTension ? "Tension" : "Belief"} (${d.group}):\n\n${d.label}`);
    })
    .call(drag(simulation));

  const label = g.selectAll(".label")
    .data(nodes, d => d.id)
    .join("text")
    .attr("class", "label")
    .attr("text-anchor", "middle")
    .attr("dy", "-0.7em")
    .attr("fill", "#ccc")
    .style("font-size", "10px")
    .text(d => d.label.length > 30 ? d.label.slice(0, 28) + "…" : d.label);

  simulation.nodes(nodes);
  simulation.force("link").links(links);

  for (let i = 0; i < 50; i++) simulation.tick();

  simulation.on("tick", () => {
    node
      .attr("cx", d => d.x = Math.max(padding, Math.min(width - padding, d.x)))
      .attr("cy", d => d.y = Math.max(padding, Math.min(height - padding, d.y)));

    label
      .attr("x", d => d.x)
      .attr("y", d => d.y);

    link
      .attr("x1", d => d.source.x)
      .attr("y1", d => d.source.y)
      .attr("x2", d => d.target.x)
      .attr("y2", d => d.target.y);
  });

  simulation.alpha(1).restart();
}

function drag(simulation) {
  return d3.drag()
    .on("start", (event, d) => {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    })
    .on("drag", (event, d) => {
      d.fx = event.x;
      d.fy = event.y;
    })
    .on("end", (event, d) => {
      d.fx = Math.max(padding, Math.min(width - padding, d.fx));
      d.fy = Math.max(padding, Math.min(height - padding, d.fy));
      if (!event.active) simulation.alphaTarget(0);
    });
}

function fadeOutResolvedNodes(resolvedIds) {
  if (!resolvedIds?.length) return;

  nodes.forEach(n => {
    resolvedIds.forEach(id => {
      if (n.id.includes(id)) n.resolved = true;
    });
  });

  const g = svg.select("#graph-group");

  g.selectAll(".node")
    .filter(d => d.resolved)
    .transition()
    .duration(1500)
    .style("opacity", 0)
    .remove();

  g.selectAll(".link")
    .filter(d => d.source.resolved || d.target.resolved)
    .transition()
    .duration(1500)
    .style("opacity", 0)
    .remove();

  g.selectAll(".label")
    .filter(d => d.resolved)
    .transition()
    .duration(1500)
    .style("opacity", 0)
    .remove();

  nodes = nodes.filter(n => !n.resolved);
  links = links.filter(l => !(l.source.resolved || l.target.resolved));

  simulation.nodes(nodes);
  simulation.force("link").links(links);
  simulation.alpha(0.5).restart();
}

window.onload = () => {
  setTimeout(() => {
    initThoughtMapViz("#thoughtmap");

    // Optional hooks
    if (typeof loadThoughtMap === "function") loadThoughtMap();
    if (typeof loadHistory === "function") loadHistory();
  }, 300);
};

// Optional: Expose simulation controls
window.pauseMap = () => simulation.stop();
window.resumeMap = () => simulation.alpha(1).restart();
