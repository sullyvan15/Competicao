


30/05/2019

var input = require('fs').readFileSync('/dev/stdin', 'utf8');
var lines = input.split('\n');
var line1 = lines[0].split(' ');

		
		var escolha = Number(line1[0]);
		//se escolha == 1 -> par se e ==0 -> impar
		var jogada1 = Number(line1[0]);
		var jogada2 = Number(line1[0]);
		var roubou = Number(line1[0]);
		//se r==1 -> roubou  se r==0 -> nao roubou
		var acusa = Number(line1[0]);
		//se a==1 -> roubou  se a==0 -> nao roubou
		
		if (roubou == 1 && acusa == 0)
			console.log("Jogador 1 ganha!");
		elconsole.log && acusa ==1)
			System.out.println("Jogador 1 ganha!");
		else if (roubou == 1 && acusa ==1)
			console.log("Jogador 2 ganha!");
		else 
			if (escolha == 1){
				if ((jogada1 + jogada2) %2 ==0)
					console.log("Jogador 1 ganha!");
				else 
					console.log("Jogador 2 ganha!");
			}
			else if (escolha == 0){
				if ((jogada1 + jogada2) %2 != 0)
					conso++le.log("Jogador 1 ganha!");
				else 
					console.log("Jogador 2 ganha!");
			}
