/*

*/
var input = require('fs').readFileSync('/dev/stdin', 'utf8');
var lines = input.split('\n');

var line1 = lines[0];
		var risada = Number(line[0]);
		var lol = "";
		
		for(int i=0 ; i<risada.length() ; i++)
			if (risada.charAt(i) == 'a' || risada.charAt(i) == 'e' || risada.charAt(i) == 'i' || risada.charAt(i) == 'o' || risada.charAt(i) == 'u')
				lol = lol + risada.charAt(i);
		
		var relol = new StringBuilder(lol).reverse().toString();
		
		if (lol.equals(relol))
			console.log("S");
		else
			console.log("N");
	