#include<iostream>
#include<algorithm>
#include<cmath>
#include<cstring>
#include<cstdio>
#include<queue>
using namespace std;
long long nbytes=100000000;
int main(){
	//freopen(".in","r",stdin);
	freopen("input.txt","w",stdout);
	for(int i=0;i<nbytes;i++)
	{
		cout<<char(i%26+65);
	}
	return 0;
}

