create_graph_dw = (data, colors, directed, weights, target) => {
    const width = 750;
    const height = 450;
    const nodes = Array.from(new Set(data.flatMap(l => [l.source, l.target])), id => ({'id':id, 'color':colors[id]}));
    const links = data.map(d => Object.create(d));

    drag = simulation => {
        function dragstarted(event, d) {
            if (!event.active) simulation.alphaTarget(0.3).restart();
            d.fx = d.x;
            d.fy = d.y;
        }
        
        function dragged(event, d) {
            d.fx = event.x;
            d.fy = event.y;
        }
        
        function dragended(event, d) {
            if (!event.active) simulation.alphaTarget(0);
            d.fx = null;
            d.fy = null;
        }
        
        return d3.drag()
            .on('start', dragstarted)
            .on('drag', dragged)
            .on('end', dragended);
    }
      
    const simulation = d3.forceSimulation(nodes)
        .force('link', d3.forceLink(links).id(d => d.id).distance(150))
        .force('charge', d3.forceManyBody().strength(-400))
        .force('x', d3.forceX())
        .force('y', d3.forceY());
    
    const svg = d3.select(target).append('svg')
        .attr('viewBox', [-width / 2, -height / 2, width, height])
        .attr('width', width)
        .attr('height', height)
        .attr('style', 'max-width: 100%; height: auto; font: 12px sans-serif;');
    var link;   
    if (directed==1) {
        svg.append('defs').selectAll('marker')
        .data(links)
        .join('marker')
            .attr('id', 'arrow')
            .attr('viewBox', '0 -5 10 10')
            .attr('refX', 25)
            .attr('refY', -0.5)
            .attr('markerWidth', 6)
            .attr('markerHeight', 6)
            .attr('orient', 'auto')
        .append('path')
            .attr('fill', 'orange')
            .attr('d', 'M0,-5L10,0L0,5');
    
        link = svg.append('g')
            .attr('fill', 'none')
            .attr('stroke-width', 1.5)
        .selectAll('path')
        .data(links)
        .join('path')
            .attr('stroke', 'orange')
            .attr('id', function (d, i) {return 'pathx' + i})
            .attr('marker-end', `url(${new URL(`#arrow`, location)})`);    
    } else {
        link = svg.append('g')
            .attr('fill', 'none')
            .attr('stroke-width', 1.5)
        .selectAll('path')
        .data(links)
        .join('path')
            .attr('stroke', 'orange')
            .attr('id', function (d, i) {return 'pathx' + i})
    }

    const edgelabels = svg.selectAll('.edgelabel')
        .data(links)
        .enter()
        .append('text')
        .style('pointer-events', 'none')
        .attr('class', 'edgelabel')
        .attr('id', function (d, i) {return 'edgelabel' + i})
        .attr('font-size', 16)
        .attr('rotate', 0)
        .attr('fill', 'black');
    
    if (weights==1) {
        edgelabels.append('textPath') 
        .attr('xlink:href', function (d, i) {return '#pathx' + i})
        .style('pointer-events', 'none')
        .attr('startOffset', '50%')
        .attr('text-anchor', 'middle')
        .text(d => (d.type));
    }
    
    const node = svg.append('g')
        .attr('fill', 'currentColor')
        .attr('stroke-linecap', 'round')
        .attr('stroke-linejoin', 'round')
    .selectAll('g')
    .data(nodes)
    .join('g')
        .call(drag(simulation));
    
    node.append('circle')
        .attr('stroke', 'black')
        .attr('fill', d => d.color)
        .attr('stroke-width', 1.5)
        .attr('r', 14);
    
    node.append('text')
        .attr('x', -4)
        .attr('y', 4)
        .text(d => d.id)
    .clone(true).lower()
        .attr('fill', 'white')
        .attr('stroke', 'white')
        .attr('stroke-width', 3);
    
    simulation.on('tick', () => {
        link.attr('d', linkArc);
        node.attr('transform', d => `translate(${d.x},${d.y})`);
        edgelabels.attr('rotate', d => (d.target.x>d.source.x) ? 0 : 180)
        if (weights==1) {
            edgelabels.selectAll('textPath').text(d => (d.target.x>d.source.x) ? d.type : d.type.split('').reverse().join(''))
        }
    });
    
    function linkArc(d) {
        const r = Math.hypot(d.target.x - d.source.x, d.target.y - d.source.y);
        return `
            M${d.source.x},${d.source.y}
            A${r},${r} 0 0,1 ${d.target.x},${d.target.y}
        `;
    }
}

create_histogram = (hdata, target) => {
    // Specify the chart’s dimensions.
    const width = 750;
    const height = 450;
    const marginTop = 40;
    const marginRight = 0;
    const marginBottom = 40;
    const marginLeft = 40;

    // Create the horizontal scale and its axis generator.
    const x = d3.scaleBand()
        .domain(d3.sort(hdata, d => d.energy).map(d => d.energy))
        .range([marginLeft, width - marginRight])
        .padding(0.1);

    const xAxis = d3.axisBottom(x).ticks(100).tickSizeOuter(0);


    // Create the vertical scale.
    const y = d3.scaleLinear()
        .domain([0, d3.max(hdata, d => d.num_occurrences)]).nice()
        .range([height - marginBottom, marginTop]);

    // Create the SVG container and call the zoom behavior.
    const svg = d3.select(target).append('svg')
        .attr('viewBox', [0, 0, width, height])
        .attr('width', width)
        .attr('height', height)
        .attr('style', 'max-width: 100%; height: auto;')
        .call(zoom);

    // Append the bars.
    svg.append('g')
        .attr('class', 'bars')
        .attr('fill', '#fca311')
    .selectAll('rect')
    .data(hdata)
    .join('rect')
        .attr('x', d => x(d.energy))
        .attr('y', d => y(0))
        .attr('height', d => 0)
        .attr('width', x.bandwidth());

    // Append the axes.
    svg.append('g')
        .attr('class', 'x-axis')
        .attr('transform', `translate(0,${height - marginBottom})`)
        .call(xAxis)
        .call((g) => g.append('text')
            .attr('x', width)
            .attr('y', marginBottom-5)
            .attr('fill', 'currentColor')
            .attr('text-anchor', 'end')
            .text('Energy level')
            .attr('font-size', 16));

    svg.append('g')
        .attr('class', 'y-axis')
        .attr('transform', `translate(${marginLeft},0)`)
        .call(d3.axisLeft(y))
        .call(g => g.select('.domain').remove())
        .call((g) => g.append('text')
            .attr('x', -marginLeft)
            .attr('y', 20)
            .attr('fill', 'currentColor')
            .attr('text-anchor', 'start')
            .text('Number of occurences')
            .attr('font-size', 16));

    function zoom(svg) {
        const extent = [[marginLeft, marginTop], [width - marginRight, height - marginTop]];

        svg.call(d3.zoom()
            .scaleExtent([1, 8])
            .translateExtent(extent)
            .extent(extent)
            .on('zoom', zoomed));

        function zoomed(event) {
            x.range([marginLeft, width - marginRight].map(d => event.transform.applyX(d)));
            svg.selectAll('.bars rect').attr('x', d => x(d.energy)).attr('width', x.bandwidth());
            svg.selectAll('.x-axis').call(xAxis);
        }
    }

    svg.selectAll("rect")
        .transition()
        .duration(1800)
        .attr('y', d => y(d.num_occurrences))
        .attr('height', d => y(0) - y(d.num_occurrences))

    // TOOLTIP
    const tooltip = d3
        .select("body")
        .append("div")
        .attr("class", "svg-tooltip")
        .style("position", "absolute")
        .style("visibility", "hidden");
    
    d3.selectAll("rect")
    .on("mouseover", function(event, d) {
        // change the selection style
        d3.select(this)
        .attr('stroke-width', '2')
        .attr("stroke", "black");
        // make the tooltip visible and update its text
        tooltip
        .style("visibility", "visible")
        .text(`Energy: ${d.energy}, Occurences: ${d.num_occurrences}`);
    })
    .on("mousemove", function(event,d) {
        tooltip
        .style("top", event.y - 10 + "px")
        .style("left", event.x + 10 + "px");
    })
    .on("mouseout", function() {
        // change the selection style
        d3.select(this).attr('stroke-width', '0');

        tooltip.style("visibility", "hidden");
    });
}