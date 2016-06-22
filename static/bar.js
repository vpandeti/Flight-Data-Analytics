var width = 1000, height = 450, padding = 20;
var margin = { top: 0, right: 30, bottom: 20, left:50 };
var fontFamily = 'verdana';
var barFillColor = 'steelblue';
var barFocusColor = 'yellow';
var strokeColor = '#F00226';
var noOfBins = 32;
var xAxisPadding = 1;
var toolTipBackground = '#FFF';
var binVariables = [0,1,2], chartIndex = 0;
var pieTextAlign = 'middle', pieWidth = 500, pieHeight = 500, pieRadius = 250;
var pieDataColor = '#FFF';
var xAxisType;

function drawScatter(sData, rs) {
    // var data = $.map(sData, function(el) { return el });
    d3.select('#chart').remove();
    var data = JSON.parse(sData),
        array = []
    var min = 0, max = 0
    for(var i=0; i< Object.keys(data[0]).length; ++i){
        obj = {}
        obj.x = data[0][i];
        obj.y = data[1][i];
        obj.clusterid = data['clusterid'][i]
        obj.arrival = data['arrival'][i]
        obj.departure = data['departure'][i]
        array.push(obj);
    }
    data = array;

    var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = 960 - margin.left - margin.right,

    height = 500 - margin.top - margin.bottom;

    var xValue = function(d) { return d.x;}, xScale = d3.scale.linear().range([0, width]),
        xMap = function(d) { return xScale(xValue(d));}, xAxis = d3.svg.axis().scale(xScale).orient("bottom");

    var yValue = function(d) { return d.y;}, yScale = d3.scale.linear().range([height, 0]),
        yMap = function(d) { return yScale(yValue(d));}, yAxis = d3.svg.axis().scale(yScale).orient("left");

    var cValue
    if(rs) {
        cValue = function(d) { return d.clusteridx;}
    } else {
        cValue = function(d) { return d.clusterid;}
    }
    var color = d3.scale.category10();

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var tooltip = d3.select("body").append('div').style('position','absolute');

    xScale.domain([d3.min(data, xValue)-1, d3.max(data, xValue)+1]);
    yScale.domain([d3.min(data, yValue)-1, d3.max(data, yValue)+1]);

    svg.append("g")
          .attr("transform", "translate(0," + height + ")")
          .attr("class", "x axis")
          .call(xAxis)
        .append("text")
          .attr("class", "label")
          .attr("y", -6)
          .attr("x", width)
          .text("x")
          .style("text-anchor", "end");

    svg.append("g")
          .attr("class", "y axis")
          .call(yAxis)
        .append("text")
          .attr("class", "label")
          .attr("y", 6)
          .attr("transform", "rotate(-90)")
          .attr("dy", ".71em")
          .text("y")
          .style("text-anchor", "end");

    svg.selectAll(".dot")
          .data(data)
          .enter().append("circle")
          .attr("class", "dot")
          .attr("cx", xMap)
          .attr("r", 3.5)
          .attr("cy", yMap)
          .style("fill", function(d) { return color(cValue(d));})
          .on("mouseover", function(d) {
              tooltip.transition().style('opacity', .9).style(
							'font-family', fontFamily).style('color','steelblue')
              tooltip.html("Departure: " + d.departure + ", Arrival: " + d.arrival)
                   .style("top", (d3.event.pageY - 28) + "px")
                   .style("left", (d3.event.pageX + 5) + "px");
          })
          .on("mouseout", function(d) {
              tooltip.transition()
                   .duration(500)
                   .style("opacity", 0);
              tooltip.html('');
          });
}

function drawLSA(sData, rs) {
    d3.select('#chart').remove();
    var data = JSON.parse(sData),
        array = [];
    var min = 0, max = 0;
    var noClusters = data.length;

    var margin = {top: 20, right: 20, bottom: 30, left: 40};
    var cR = 100;

    var color = ['steelblue', '#E69F9F', '#88C37C', '#EEDDCC'];

    var svg = d3.select("body").append("svg")
        .attr('id', 'chart')
        .attr("height", height + margin.top + margin.bottom)
        .attr("width", width + margin.left + margin.right)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    var tooltip = d3.select("body").append("div")
        .attr("class", "tooltip")
        .style("opacity", 0);

    var item = [];
    var start = cR, end, unit = width/noClusters/2;
    for (var i = 0; i < noClusters; i++) {

        // var randX = Math.floor((Math.random() * 200) + (i)*cR));
        randX = start;
        var randY = Math.floor((Math.random() * (300 - cR)) + cR);
        start += unit + cR;
        // draw dots
        svg.append("circle")
          .attr("class", "dot")
          .attr("r", cR)
          .attr("cx", randX)
          .attr("cy", randY)
          .style("fill", color[i])
          .on("mouseover", function(d) {
              tooltip.transition()
                   .duration(200)
                   .style("opacity", .9);
              tooltip.html(d.x)
                   .style("left", (d3.event.pageX + 5) + "px")
                   .style("top", (d3.event.pageY - 28) + "px");
          })
          .on("mouseout", function(d) {
              tooltip.transition()
                   .duration(500)
                   .style("opacity", 0);
          });

          item = data[i];
          for(var j = 0; j < item.length; j++) {
            var v1X = (randX + cR), v1Y = (randX - cR);
            var v2X = (randY + cR), v2Y = (randY - cR);
            var rX = Math.random() * (v1X - v1Y) + v1Y;
            var rY = Math.random() * (v2X - v2Y) + v2Y;
            svg.append("text")
                .attr("x", rX)
                .attr("y", rY)
                .attr("dy", ".35em")
                .text(item[j])
                .style('font', '11px arial');
          }

    }

    var xValue = function(d) { return d.x;}, xScale = d3.scale.linear().range([0, width]),
        xMap = function(d) { return xScale(xValue(d));}, xAxis = d3.svg.axis().scale(xScale).orient("bottom");

    var yValue = function(d) { return d.y;}, yScale = d3.scale.linear().range([height, 0]),
        yMap = function(d) { return yScale(yValue(d));}, yAxis = d3.svg.axis().scale(yScale).orient("left");

    svg.append("g")
      .attr("transform", "translate(0," + height + ")")
      .call(xAxis)
      .append("text")
      .attr("class", "label")
      .attr("x", width)
      .attr("y", -6)
      .text("x")
      .style("text-anchor", "end");

    svg.append("g")
          .call(yAxis)
          .append("text")
          .attr("class", "label")
          .attr("transform", "rotate(-90)")
          .attr("y", 6)
          .attr("dy", ".71em")
          .text("y")
          .style("text-anchor", "end");
}

function drawScreePlot(sData) {
    var d = JSON.parse(sData)
    var obj;
    var data = [];
    for (var i = 0; i < 2; i++) {
      obj = {};
      obj.x = i;
      obj.y = d['variance'][i];
      data.push(obj);
    }

    var screeMargin = {top: 70, right: 40, bottom: 30, left: 70},
        screeWidth = 300 - screeMargin.left - screeMargin.right,
        screeHeight = 300 - screeMargin.top - screeMargin.bottom;

    var screeSvg = d3.select("body").append("svg")
        .attr('id', 'scree')
        .attr("height", screeHeight + screeMargin.top + screeMargin.bottom)
        .attr("width", screeWidth + screeMargin.left + screeMargin.right)
        .append("g")
        .attr("transform", "translate(" + screeMargin.left + "," + screeMargin.top + ")");

    var y = d3.scale.linear().range([screeHeight, 0])
    var x = d3.scale.ordinal().rangeRoundBands([0, screeWidth], .05);

    var xAxis = d3.svg.axis()
    .scale(x)
    .orient("bottom")

    var yValue = function(d) { return d.y;}, yScale = d3.scale.linear().range([height, 0]),
    yMap = function(d) { return yScale(yValue(d));}, yAxis = d3.svg.axis().scale(yScale).orient("left");

    y.domain([0, d3.max(data, function(d) { return d.y; })]);
    x.domain(data.map(function(d) { return d.x; }));

    screeSvg.selectAll("bar")
        .data(data)
        .enter().append("rect")
        .style("margin-left", "2px")
        .style("fill", "steelblue")
        .attr("x", function(d) { return x(d.x); })
        .attr("y", function(d) { return y(d.y); })
        .attr("width", x.rangeBand())
        .attr("height", function(d) { return screeHeight - y(d.y); });

    screeSvg.append("g")
        .attr("transform", "translate(0," + screeHeight + ")")
        .attr("class", "x axis")
        .call(xAxis)
        .selectAll("text")
        .attr("dy", "-.55em")
        .style("text-anchor", "end")
        .attr("dx", "-.8em")
        .attr("transform", "rotate(-90)" );

    screeSvg.append("g")
        .attr("class", "y axis")
        .call(yAxis)
      .append("text")
        .attr("y", 6)
        .attr("transform", "rotate(-90)")
        .style("text-anchor", "end")
        .attr("dy", ".71em")
        .text("Variance");
}

function mapSelect() {
    var dropdown = document.getElementById("map");
    var selectedValue = dropdown.options[dropdown.selectedIndex].value;
    if(selectedValue == -1) {
        // Do nothing
    } else if(selectedValue == "PCA_RANDOM_SAMPLING") {
        get_map('/pca_random', true, false, true);
    } else if(selectedValue == "PCA_ADAPTIVE_SAMPLING") {
        get_map('/pca_adaptive', false, false, true);
    } else if(selectedValue == "ISOMAP_RANDOM_SAMPLING") {
        get_map('/isomap_random', true, false, false);
    } else if(selectedValue == "ISOMAP_ADAPTIVE_SAMPLING") {
        get_map('/isomap_adaptive', false, false, false);
    } else if(selectedValue == "MDS_EUCLIDEAN_RANDOM_SAMPLING") {
        get_map('/mds_euclidean_random', true, false, false);
    } else if(selectedValue == "MDS_EUCLIDEAN_ADAPTIVE_SAMPLING") {
        get_map('/mds_euclidean_adaptive', false, false, false);
    } else if(selectedValue == "MDS_COSINE_RANDOM_SAMPLING") {
        get_map('/mds_cosine_random', true, false, false);
    } else if(selectedValue == "MDS_COSINE_ADAPTIVE_SAMPLING") {
        get_map('/mds_cosine_adaptive', false, false, false);
    } else if(selectedValue == "MDS_CORRELATION_RANDOM_SAMPLING") {
        get_map('/mds_correlation_random', true, false, false);
    } else if(selectedValue == "MDS_CORRELATION_ADAPTIVE_SAMPLING") {
        get_map('/mds_correlation_adaptive', false, false, false);
    } else if(selectedValue == "LATENT_SEMANTIC_ANALYSIS") {
        get_map('/lsa', false, true, false);
    }
    d3.select('#scree').remove();
}

function get_map(url, rs, lsa, isPCA) {
	$.ajax({

	  type: 'GET',
	  url: url,
      contentType: 'application/json; charset=utf-8',
	  xhrFields: {
		withCredentials: false
	  },
	  headers: {

	  },
	  success: function(result) {
	    if(lsa) {
		    drawLSA(result, rs);
		} else {
		    drawScatter(result, rs);
		    if(isPCA) {
		        drawScreePlot(result)
		    }
		}
	  },
	  error: function(result) {
		$("#body1").html(result);
	  }
	});
}