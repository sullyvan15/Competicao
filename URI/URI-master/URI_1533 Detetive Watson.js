/*

*/
var input = require('fs').readFileSync('/dev/stdin', 'utf8');
var lines = input.split('\n');

var line1 = lines[0];
		
		var n = Number(line1[0]);

		while(n!=0){
			
					
			for (var i=0 ; i<n ; i++)
				x[i] = sc.nextInt();

			var Maior = x[0];
			var posMaior = 0; 
			for (var i=1 ; i<n ; i++)
				if (Maior<=x[i]);
					Maior=x[i];
					posMaior=i;
				
			
				
			Maior = x[0];
			posMaior = 0; 
			for (int i=1 ; i<n ; i++)
				if (Maior<=x[i]);
					Maior=x[i];
					posMaior=i;
				;
			
			console.log(posMaior+1);
			
