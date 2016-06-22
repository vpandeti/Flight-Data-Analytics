var monthMap = {
	"0" : "Jan",
	"1" : "Feb",
	"2" : "Mar",
	"3" : "Apr",
	"4" : "May",
	"5" : "Jun",
	"6" : "Jul",
	"7" : "Aug",
	"8" : "Sep",
	"9" : "Oct",
	"10" : "Nov",
	"11" : "Dec"
}

var panel = [ {
	id : "1",
	category : "comedy",
	channel : "The Tonight Show Starring Jimmy Fallon"
}, {
	id : "2",
	category : "comedy",
	channel : "CollegeHumor"
}, {
	id : "3",
	category : "comedy",
	channel : "Jimmy Kimmel Live"
}, {
	id : "4",
	category : "comedy",
	channel : "Just For Laughs Gags"
}, {
	id : "5",
	category : "comedy",
	channel : "Comedy Central"
}, {
	id : "6",
	category : "travel",
	channel : "FunForLouis"
}, {
	id : "7",
	category : "travel",
	channel : "BBC Earth"
}, {
	id : "8",
	category : "travel",
	channel : "Africa Travel Channel"
}, {
	id : "9",
	category : "travel",
	channel : "BBC Travel Show"
}, {
	id : "10",
	category : "travel",
	channel : "Samuel & Audrey"
}, {
	id : "11",
	category : "automobiles",
	channel : "Ferrari"
}, {
	id : "12",
	category : "automobiles",
	channel : "Mercedes-Benz"
}, {
	id : "13",
	category : "automobiles",
	channel : "Automobile Magazine"
}, {
	id : "14",
	category : "automobiles",
	channel : "Motor Trend Channel"
}, {
	id : "15",
	category : "automobiles",
	channel : "BMW"
}, {
	id : "16",
	category : "music",
	channel : "JustinBieberVEVO"
}, {
	id : "17",
	category : "music",
	channel : "RihannaVEVO"
}, {
	id : "18",
	category : "music",
	channel : "OneDirectionVEVO"
}, {
	id : "19",
	category : "music",
	channel : "TaylorSwiftVEVO"
}, {
	id : "20",
	category : "music",
	channel : "KatyPerryVEVO"
}, {
	id : "21",
	category : "news",
	channel : "BBC News"
}, {
	id : "22",
	category : "news",
	channel : "VICE News"
}, {
	id : "23",
	category : "news",
	channel : "Sky News"
}, {
	id : "24",
	category : "news",
	channel : "DNews"
}, {
	id : "25",
	category : "news",
	channel : "Forbes"
}, {
	id : "26",
	category : "education",
	channel : "Google for Education"
}, {
	id : "27",
	category : "education",
	channel : "Khan Academy"
}, {
	id : "28",
	category : "education",
	channel : "Discovery Education"
}, {
	id : "29",
	category : "education",
	channel : "TED-Ed"
}, {
	id : "30",
	category : "education",
	channel : "SciShow"
} ];

var view = [ {
	id : "V1",
	view : "Videos Uploaded Vs Time"
}, {
	id : "V2",
	view : "Views Vs Time"
}, ];

var loadcsv = [ {
	viewid : "V1",
	channelid : "1",
	csv : "V1_1.csv"
}, ];

function getView() {

}

function getChannel(id) {
	for (var i = 0; i < panel.length; i++) {
		if (panel[i].id == id) {
			return panel[i].channel
		}
	}
	return panel[0].channel;
}

function getCategory(id) {
	for (var i = 0; i < panel.length; i++) {
		if (panel[i].category == id) {
			return panel[i].category
		}
	}
	return panel[0].category;
}