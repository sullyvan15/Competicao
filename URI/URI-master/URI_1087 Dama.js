/*

*/
var input = require('fs').readFileSync('/dev/stdin', 'utf8');
var lines = input.split('\n');

var line1 = lines[0];
		

		var x1,y1,x2,y2;
		//se estiver na mesma linha ou mesma coluna ou mesma diagonal, gasta 1 movimento
		//se estiver em qualquer outra posição, a rainha gastará 2 movimentos!
	    while(true){
	    	x1 = Number(line1[0]);
	    	y1 = Number(line1[1]);
	    	x2 = Number(line1[2]);
	    	y2 = Number(line1[3]);
	    	if(x1 == 0 && y1 == 0 && x2 == 0 && y2 == 0) break; //condição de parada
	    	if(x1 == x2 && y1 == y2)		//mesma posição
	    		console.log("0");
	        else if(x1 == x2 || y1 == y2)	//mesma linha ou mesma coluna
	        	console.log("1");
	    	else if((x2-x1)==-(y2-y1) || -(x2-x1)==-(y2-y1) || -(x2-x1)==(y2-y1) ||  (x2-x1)==(y2-y1))
	    		console.log("1");	//mesma diagonal
	        else 
	        	console.log("2");	//o resto...
	    }
	