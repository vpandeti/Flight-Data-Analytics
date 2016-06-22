var MONTHLY_PERIOD = 1, QUARTERLY_PERIOD = 2, HALF_YEARLY_PERIOD = 3, YEARLY_PERIOD = 4;
var MONTHLY_PERIOD_VALUE = 30, QUARTERLY_PERIOD_VALUE = 90, HALF_YEARLY_PERIOD_VALUE = 180, YEARLY_PERIOD_VALUE = 365;
var period = MONTHLY_PERIOD;
var response, data = [];
var monthly = [], quarterly = [], halfyearly = [], yearly = [], startdate, enddate, statistics;

function addDays(date, days) {
	var result = new Date(date);
	result.setDate(result.getDate() + days);
	return monthMap[result.getUTCMonth()] + " " + result.getFullYear();
}

function monthlyTimePeriodClick() {
	$("#divTimeFrame input").css('background-color', 'white')
	$('#monthlyButton').css('background-color', '#ffa500')
	period = MONTHLY_PERIOD;
	assignData();
	drawVideoCountGraph();
}

function quarterlyTimePeriodClick() {
	$("#divTimeFrame input").css('background-color', 'white')
	$('#quarterlyButton').css('background-color', '#ffa500')
	period = QUARTERLY_PERIOD;
	assignData();
	drawVideoCountGraph();
}

function halfYearlyTimePeriodClick() {
	$("#divTimeFrame input").css('background-color', 'white')
	$('#halfyearlyButton').css('background-color', '#ffa500')
	period = HALF_YEARLY_PERIOD;
	assignData();
	drawVideoCountGraph();
}

function yearlyTimePeriodClick() {
	$("#divTimeFrame input").css('background-color', 'white')
	$('#yearlyButton').css('background-color', '#ffa500')
	period = YEARLY_PERIOD
	assignData();
	drawVideoCountGraph();
}

function getTimePeriodValue() {
	if (period == MONTHLY_PERIOD)
		return MONTHLY_PERIOD_VALUE;
	else if (period == QUARTERLY_PERIOD)
		return QUARTERLY_PERIOD_VALUE;
	else if (period == HALF_YEARLY_PERIOD)
		return HALF_YEARLY_PERIOD_VALUE;
	else if (period == YEARLY_PERIOD)
		return YEARLY_PERIOD_VALUE;
}

function assignData() {
	if (period == MONTHLY_PERIOD)
		data = monthly;
	else if (period == QUARTERLY_PERIOD)
		data = quarterly;
	else if (period == HALF_YEARLY_PERIOD)
		data = halfyearly;
	else if (period == YEARLY_PERIOD)
		data = yearly;
}

function videoCount(respText) {
	var json = respText;
	var obj = JSON.parse(json);
	startdate = obj.startdate, enddate = obj.enddate,
			statistics = obj.statistics;
	var stats;
	for (var i = 0; i < statistics.length; i++) {
		stats = statistics[i];
		if (stats.timeframe == 30)
			monthly = stats.count;
		else if (stats.timeframe == 90)
			quarterly = stats.count;
		else if (stats.timeframe == 180)
			halfyearly = stats.count;
		else
			yearly = stats.count;
	}
	data = monthly;
	drawVideoCountGraph();
}

function drawVideoCountGraph() {
	$("#chart").html("");
	$("#trendFrame").css('display', 'none');
	$("#divTimeFrame").css('display', "block");
	var margin = {
		top : 20,
		right : 20,
		bottom : 70,
		left : 40
	}, width = 800 - margin.left - margin.right, height = 450 - margin.top
			- margin.bottom;
	var x = d3.scale.ordinal().rangeRoundBands([ 0, width ], .05);
	var y = d3.scale.linear().range([ height, 0 ]);
	var xAxis = d3.svg.axis().scale(x).orient("bottom");
	var yAxis = d3.svg.axis().scale(y).orient("left").ticks(10);

	var svg = d3.select("#chart").append("svg").attr("width",
			width + margin.left + margin.right).attr("height",
			height + margin.top + margin.bottom).style('padding-left', '20px')
			.style('padding-bottom', '20px').append("g").attr("transform",
					"translate(" + margin.left + "," + margin.top + ")");

	x.domain(data.map(function(d, i) {
		return i;
	}));
	y.domain([ 0, d3.max(data, function(d) {
		return d;
	}) ]);

	var tip = d3.tip().attr('class', 'd3-tip').offset([ -10, 0 ]).html(
			function(d) {
				return "<strong>Videos:</strong> <span style='color:red'>" + d
						+ "</span>";

			});
	svg.call(tip);

	svg.append("g").attr("class", "x axis").attr("transform",
			"translate(0," + height + ")").call(xAxis).selectAll("text").style(
			"text-anchor", "end").style('margin-top', '10px').attr("transform",
			"rotate(-45)").attr("dx", "-.8em").attr("dy", "-.55em").text(
			function(d, i) {
				return addDays(startdate, i * getTimePeriodValue(period));
			});

	svg.append("g").attr("class", "y axis").call(yAxis).append("text").attr(
			"transform", "rotate(-90)").attr("y", -50).attr("dy", ".71em")
			.style("text-anchor", "end").text("No of Videos");

	svg.selectAll("bar").data(data).enter().append("rect").style("fill",
			"steelblue").attr("x", function(d, i) {
		return x(i);
	}).attr("width", x.rangeBand()).attr("y", function(d) {
		return height;
	}).attr("height", 0).on('mouseover', function(d) {
		tip.show(d);
		d3.select(this).style('fill', "#e6b000");
	}).on('mouseout', function(d) {
		tip.hide(d);
		d3.select(this).style('fill', 'steelblue');
	}).transition().duration(1000).delay(200).attr('height',
			function(d) {
				return height - y(d);
			}).attr('y', function(d) {
		return y(d);
	});

}