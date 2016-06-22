var selectedTrend = 'views'
function channelTrend(obj)
{
	$("#divTimeFrame").css('display',"none");
	var jsobj = JSON.parse(obj);
	var categoryView = false;
	if (jsobj.length == 1)
		jsobj = jsobj[0]
	else
		categoryView = true;
	var text =  jsobj.title + " channel has " + jsobj.subscribers + " subscribers, "
	          + jsobj.views + " views, and "
	          + jsobj.videos + " videos ";
	$("#chart").html(text);
	data = jsobj;	
	if (categoryView)
	{
		$("#trendFrame").css('display','none');
		loadCategoryView(data)
	}
	else
	{
		loadChannelView(data.statistics, selectedTrend);
	}
}

function process(view)
{
	selectedTrend = view;
	loadChannelView(data.statistics, view);
}

var legendFilterChannel = {
		"views":{"val":true,"cls":"showlegend"},
		"likes":{"val":true,"cls":"showlegend"},
		"dislikes":{"val":true,"cls":"showlegend"},
		"comments":{"val":true,"cls":"showlegend"},
		"date":{"val":true,"cls":"showlegend"},
}

function loadChannelViewBetter(dataoriginal)
{
	var data = jQuery.extend(true, [], dataoriginal);	
	$("#chart").html("");
	var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = $("#chart").width() - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;
	var svg = d3.select("#chart").append("svg")
	 .attr("width", width + margin.left + margin.right)
	 .attr("height", height + margin.top + margin.bottom)
	 .append("g")
	 .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	
	var chart = d3LineWithLegend()
               .xAxis.label('Days')
               .yAxis.label('count');
	var svg = d3.select('#chart svg')
     		  .datum(generateData(data))

 chart.dispatch.on('showTooltip', function(e) {
	 var offset = $('#chart').offset(), // { left: 0, top: 0 }
	       left = e.pos[0] + offset.left,
	       top = e.pos[1] + offset.top,
	       formatter = d3.format(".04f");
	
	   var content = '<h3>' + e.series.label + '</h3>' +
	                 '<p>' +
	                 '<span class="value">[' + e.point[0] + ', ' + formatter(e.point[1]) + ']</span>' +
	                 '</p>';
	   nvtooltip.show([left, top], content);
	 });
	
	 chart.dispatch.on('hideTooltip', function(e) {
	   nvtooltip.cleanup();
	 });
	 

	 svg.transition().duration(500)
	     .call(chart);

}

//$(window).resize(function() {
//	   var margin = chart.margin();
//
//	   chart
//	     .width(width(margin))
//	     .height(height(margin));
//
//	   d3.select('#chart svg')
//	     .attr('width', width(margin))
//	     .attr('height', height(margin))
//	     .call(chart);
//
//	   });

function width(margin) {
	   var w = $(window).width() - 20;

	   return ( (w - margin.left - margin.right - 20) < 0 ) ? margin.left + margin.right + 2 : w;
	 }

	 function height(margin) {
	   var h = $(window).height() - 20;

	   return ( h - margin.top - margin.bottom - 20 < 0 ) ? 
	             margin.top + margin.bottom + 2 : h;
	 }


	 //data
	 function generateData(data) {
	
	   var sin = [],
	       r1 = Math.random(),
	       r2 = Math.random(),
	       r3 = Math.random(),
	       r4 = Math.random();

	   for (var i = 0; i < 100; i++) {
	     sin.push([ i, r1 * Math.sin( r2 +  i / (10 * (r4 + .5) ))]);
	   }
	   var pushdata = []
	   for (var i = 0; i < data.length; i++)
		{
		   pushdata.push([data[i].daydiff, data[i].views]);
		}
	   var retdata = [
	         	     {
	         		       data: pushdata,
	         		       label: "views"
	         		     }
	         		   ];
	   console.log(retdata)
	   return  retdata
	 }

function loadChannelView(dataoriginal, selView)
{
	$("#trendFrame").css('display','block');
	$("#trendFrame input").css('backgroundColor', 'white');
	if (selView == 'views')
		$("#viewtrend").css('backgroundColor', '#ffa500');
	if (selView == 'likes')
		$("#liketrend").css('backgroundColor', '#ffa500');
	if (selView == 'dislikes')
		$("#disliketrend").css('backgroundColor', '#ffa500');
	if (selView == 'views per like')
		$("#viewsperliketrend").css('backgroundColor', '#ffa500');
	if (selView == 'likes per dislike')
		$("#likesperdisliketrend").css('backgroundColor', '#ffa500');
	if (selView == 'duration')
		$("#durationtrend").css('backgroundColor', '#ffa500');
	var data = jQuery.extend(true, [], dataoriginal);	
	$("#chart").html("");
	var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = $("#chart").width() - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

	var x0 = d3.scale.ordinal().rangeRoundBands([0, width], .1);

	var x1 = d3.scale.ordinal();
	
	var y = d3.scale.linear().range([height, 0]);
	
	var color = d3.scale.category10();
	
	var xAxis = d3.svg.axis().scale(x0).orient("bottom");
	
	var yAxis = d3.svg.axis()
	    .scale(y).orient("left").tickFormat(d3.format(".2s"));
	
	var svg = d3.select("#chart").append("svg")
	    	 .attr("width", width + margin.left + margin.right)
	    	 .attr("height", height + margin.top + margin.bottom)
	    	 .append("g")
	    	 .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

	var parseDate = d3.time.format("%Y-%m-%d").parse;
	var bisectDate = d3.bisector(function(d) { return d.date; }).left;
	var x = d3.time.scale()
	    .range([0, width]);

	var y = d3.scale.linear()
	    .range([height, 0]);

	var color = d3.scale.category10();

	var xAxis = d3.svg.axis()
	    .scale(x)
	    .orient("bottom");

	var yAxis = d3.svg.axis()
	    .scale(y)
	    .orient("left")
	    .tickFormat(d3.format(".2s"));;

	var line = d3.svg.line()
	    .interpolate("basis")
	    .x(function(d) { return x(d.date); })
	    .y(function(d) { return y(d.counts); });

	//color.domain(d3.keys(data[0]).filter(function(key) { return key !== "date"; }));
	color.domain([selView])

	data.forEach(function(d) {
	    d.date = parseDate(d.videodate);
	    if (d.likes > 0)
	    	d['views per like'] = d.views/d.likes;
	    else
	    	d['views per like'] = 0;
	    if (d.dislikes > 0)
	    	d['likes per dislike'] = d.likes/d.dislikes;
	    else
	    	d['likes per dislike'] = 0;
	});
	if (selView == 'likes per dislike' || selView == 'views per like' || selView == 'duration')
	{
		var totalA = 0
		var totalB = 0
		var valid = 0
		var totalC = 0;
		for (var i = 0; i < data.length; i++)
		{
			if (data[i]['likes'] > 0 && data[i]['dislikes'] > 0)
			{
				totalA += data[i]['likes per dislike']
				totalB += data[i]['views per like']
				valid++;
			}
			totalC += data[i]['duration']
		}
		var likeDislikeRatio = Math.round(totalA/valid);
		var viewLikeRatio =Math.round(totalB/valid);
		var dura = Math.round(totalC/data.length);
		if (selView == 'likes per dislike' )
			$("#ratio").html('Average Ratio : ' +likeDislikeRatio);
		else if (selView == 'views per like' )
			$("#ratio").html('Average Ratio : ' +viewLikeRatio);
		else
			$("#ratio").html('Average video duration : '  + dura + ' seconds');
	}
	else {
		$("#ratio").html('');
	}
	
	  var variations = color.domain().map(function(name) {
	    return {
	      name: name,
	      values: data.map(function(d) {
	        return {date: d.date, counts: +d[name]};
	      })
	    };
	  });

	  x.domain(d3.extent(data, function(d) { return d.date; }));

	  y.domain([
	    d3.min(variations, function(c) { return d3.min(c.values, function(v) { return v.counts; }); }),
	    d3.max(variations, function(c) { return d3.max(c.values, function(v) { return v.counts; }); })
	  ]);

	  svg.append("g")
	      .attr("class", "x axis")
	      .attr("transform", "translate(0," + height + ")")
	      .call(xAxis);

	  svg.append("g")
	      .attr("class", "y axis")
	      .call(yAxis)
	    .append("text")
	      .attr("transform", "rotate(-90)")
	      .attr("y", 6)
	      .attr("dy", ".71em")
	      .style("text-anchor", "end")
	      .text("Count");

	  var city = svg.selectAll(".city")
	      .data(variations)
	    .enter().append("g")
	      .attr("class", "city");
	  
	  var focus = svg.append("g") 
	    .style("display", "none");
	  
	// append the circle at the intersection 
	    focus.append("circle")
	        .attr("class", "y")
	        .style("fill", "none")
	        .style("stroke", "blue")
	        .attr("r", 4);
	    
	 // append the rectangle to capture mouse
	    svg.append("rect")
	        .attr("width", width)
	        .attr("height", height)
	        .style("fill", "none")
	        .style("pointer-events", "all")
	        .on("mouseover", function() { focus.style("display", null); })
	        .on("mouseout", function() { focus.style("display", "none"); })
	        .on("mousemove", mousemove);

	    function mousemove() {
			var x0 = x.invert(d3.mouse(this)[0]);
			//var    i = bisectDate(variations, x0, 1),
			//    d0 = variations[i - 1],
			//    d1 = variations[i];
			//var d = x0 - d0.values.date > d1.values.date - x0 ? d1 : d0;
			d = null;
			for (var i = 0; i < variations[0].values.length; i++)
			{
				var obj = variations[0].values[i].date;
				if (obj.getYear() == x0.getYear() && obj.getMonth() == x0.getMonth() && obj.getDate() == x0.getDate())
				{
					d = variations[0].values[i];
					break;
				}
			}
			if (d != null)
				focus.select("circle.y")
			    .attr("transform",
			          "translate(" + x(d.date) + "," +
			                         y(d.counts) + ")");
	    }

	  city.append("path")
	      .attr("class", "line")
	      .attr("d", function(d) { return line(d.values); })
	      .style("stroke", function(d) { return color(d.name); });

	  city.append("text")
	      .datum(function(d) { return {name: d.name, value: d.values[d.values.length - 1]}; })
	      .attr("transform", function(d) { return "translate(" + x(d.value.date) + "," + y(d.value.counts) + ")"; })
	      .attr("x", 3)
	      .attr("dy", ".35em")
	      .text(function(d) { return d.name; });
		
}

var legendFilter = {
		"views":{"val":true,"cls":"showlegend"},
		"likes":{"val":true,"cls":"showlegend"},
		"dislikes":{"val":true,"cls":"showlegend"},
		"comments":{"val":true,"cls":"showlegend"},
		"subscribers":{"val":true,"cls":"showlegend"},
		"videos":{"val":true,"cls":"showlegend"},
		"title":{"val":false,"cls":"showlegend"}
}

function loadCategoryView(dataoriginal)
{
	var data = jQuery.extend(true, [], dataoriginal);
	$("#chart").html("");
	var margin = {top: 20, right: 20, bottom: 30, left: 40},
    width = $("#chart").width() - margin.left - margin.right,
    height = 500 - margin.top - margin.bottom;

	var x0 = d3.scale.ordinal().rangeRoundBands([0, width], .1);

	var x1 = d3.scale.ordinal();
	
	var y = d3.scale.linear().range([height, 0]);
	
	var color = d3.scale.category10();
	
	var xAxis = d3.svg.axis().scale(x0).orient("bottom");
	
	var yAxis = d3.svg.axis()
	    .scale(y).orient("left").tickFormat(d3.format(".2s"));
	
	var svg = d3.select("#chart").append("svg")
	    	 .attr("width", width + margin.left + margin.right)
	    	 .attr("height", height + margin.top + margin.bottom)
	    	 .append("g")
	    	 .attr("transform", "translate(" + margin.left + "," + margin.top + ")");
	var countNames = d3.keys(data[0]).filter(function(key) { return key != "title" });
	
	 data.forEach(function(d) {
	    d.counts = countNames.map(function(name) { return {name: name, value: +d[name]}; });
	  });
	
	  x0.domain(data.map(function(d) { return d.title; }));
	  x1.domain(countNames).rangeRoundBands([0, x0.rangeBand()]);
	  y.domain([0, d3.max(data, function(d) { return d3.max(d.counts, function(d) { if (legendFilter[d.name].val) return d.value; else return 0}); })]);

	  svg.append("g")
	      .attr("class", "x axis")
	      .attr("transform", "translate(0," + height + ")")
	      .call(xAxis);
	
	  svg.append("g")
	      .attr("class", "y axis")
	      .call(yAxis)
	    .append("text")
	      .attr("transform", "rotate(-90)")
	      .attr("y", 6)
	      .attr("dy", ".71em")
	      .style("text-anchor", "end")
	      .text("counts");
	
	  var state = svg.selectAll("#chart")
	      .data(data)
	    .enter().append("g")
	      .attr("class", "state")
	      .attr("transform", function(d) { return "translate(" + x0(d.title) + ",0)"; });
	
	  state.selectAll("rect")
	      .data(function(d) { return d.counts; })
	    .enter().append("rect")
	      .attr("width", x1.rangeBand())
	      .attr("x", function(d) { return x1(d.name); })
	      .attr("y", function(d) { if (legendFilter[d.name]['val']) return y(d.value); else return 0})
	      .attr("height", function(d) { if (legendFilter[d.name].val) return height - y(d.value);  else return 0 })
	      .style("fill", function(d) { return color(d.name); });
	
	  var legend = svg.selectAll(".legend")
	      .data(countNames.slice().reverse())
	      .enter().append("g")
	      .attr("class", function(d) { return legendFilter[d]['cls']})
	      .attr("transform", function(d, i) { return "translate(0," + i * 20 + ")"; })
	      .on("click", function(d){
	    	  if(legendFilter[d].val)
	    	  {
	    		  legendFilter[d].val = false;
	    	  	  legendFilter[d].cls = 'unselectlegend';
	    	  }
	    	  else
	    	  {
	    		  legendFilter[d].val = true;
	    		  legendFilter[d].cls = 'showlegend';
	    	  }
	    	  loadCategoryView(dataoriginal);
	      });
	
	  legend.append("rect")
	      .attr("x", width - 2)
	      .attr("width", 18)
	      .attr("height", 18)
	      .style("fill", color);
	
	  legend.append("text")
	      .attr("x", width - 8)
	      .attr("y", 9)
	      .attr("dy", ".35em")
	      .style("text-anchor", "end")
	      .text(function(d) { return d; });
}